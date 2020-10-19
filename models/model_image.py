"""
Parts of codes are from
https://github.com/RuiShu/dirt-t/codebase/models/dirtt.py
"""

import importlib

import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder, constant
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent

from data.dataset import get_attr, get_attr_image, train_image_process
from utils import accuracy, preprocessing, get_decay_var_op, get_grad_weight
from .loss import lp_loss, label_propagate, lgan_loss


def lpda_image(FLAGS):
    """
    :param FLAGS: Contains the experiment info
    :return: (TensorDict) the model
    """

    print("============================LPDA model initialization started.============================")

    nn_path = importlib.import_module("networks.{}".format(FLAGS.nn))
    exp_sz, exp_ch, nc = get_attr_image(FLAGS.nn, FLAGS.src)
    if FLAGS.nn == 'resnet':
        classifier = nn_path.ResNetModel(50, data_format="channels_last", num_classes=nc)
        feature_discriminator = nn_path.AdversarialNetwork()

    print(f"Image size: {exp_sz}, number of channels: {exp_ch}, number of classes: {nc}")

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        src_x=placeholder((None, 256, 256, exp_ch)),
        src_y=placeholder((None, nc)),
        trg_x=placeholder((None, 256, 256, exp_ch)),
        trg_y=placeholder((None, nc)),
        src_test_x=placeholder((None, exp_sz, exp_sz, exp_ch)),
        trg_test_x=placeholder((None, exp_sz, exp_sz, exp_ch)),
        src_test_y=placeholder((None, nc)),
        trg_test_y=placeholder((None, nc)),
        adpt=placeholder(None),
        src_fake=placeholder((None, exp_sz, exp_sz, exp_ch)),
        trg_fake=placeholder((None, exp_sz, exp_sz, exp_ch))
    ))

    cw = constant(1)
    dw = constant(FLAGS.dw)
    lw = constant(FLAGS.lw)
    wd = constant(FLAGS.wd)
    xw = constant(FLAGS.xw)
    sw = constant(FLAGS.sw)
    tw = constant(FLAGS.tw)

    # Preprocess images to experiment size and channels
    src_x = train_image_process(T.src_x, exp_sz, FLAGS.nn)
    trg_x = train_image_process(T.trg_x, exp_sz, FLAGS.nn)

    ##########################################
    # T.temp_src_x = src_x
    # T.temp_trg_x = trg_x
    ##########################################

    # src_test_x = preprocessing(T.src_test_x, exp_sz, exp_ch)
    # trg_test_x = preprocessing(T.trg_test_x, exp_sz, exp_ch)
    src_test_x = T.src_test_x
    trg_test_x = T.trg_test_x

    # The features and the predictions of the source and the target images
    src_p, src_e = classifier(src_x, phase=True, trim=FLAGS.trim)
    trg_p, trg_e = classifier(trg_x, phase=True, trim=FLAGS.trim)
    # src_test_e = tf.stop_gradient(nn.classifier(src_test_x, phase=True, enc_phase=True, trim=FLAGS.trim))
    # trg_test_e = tf.stop_gradient(nn.classifier(trg_test_x, phase=True, enc_phase=True, trim=FLAGS.trim))
    src_test_p, src_test_e = classifier(src_test_x, phase=False, trim=FLAGS.trim)
    trg_test_p, trg_test_e = classifier(trg_test_x, phase=False, trim=FLAGS.trim)
    # src_test_p, src_test_e = classifier(src_test_x, phase=True, trim=FLAGS.trim)
    # trg_test_p, trg_test_e = classifier(trg_test_x, phase=True, trim=FLAGS.trim)

    # src_p = nn.classifier(src_e, phase=True, enc_phase=False, trim=FLAGS.trim)
    # trg_p = nn.classifier(trg_e, phase=True, enc_phase=False, trim=FLAGS.trim, internal_update=True)

    flen = nc
    # flen = 256
    # flen = 12

    # Source true label cross entropy minimization
    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))
    loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=trg_p, logits=trg_p)) if FLAGS.xw > 0 else constant(0)

    # Adversarial domain confusion
    if FLAGS.dw > 0:
        src_logit = feature_discriminator(src_e, phase=True)
        trg_logit = feature_discriminator(trg_e, phase=True)

        loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(src_logit), logits=src_logit) +
            sigmoid_xent(labels=tf.zeros_like(trg_logit), logits=trg_logit))
        loss_dann = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(src_logit), logits=src_logit) +
            sigmoid_xent(labels=tf.ones_like(trg_logit), logits=trg_logit))
    else:
        loss_disc = constant(0)
        loss_dann = constant(0)

    # Label propagation
    if FLAGS.lw > 0:
        with tf.variable_scope('class', reuse=tf.AUTO_REUSE):
            sigma = tf.get_variable('sigma', shape=[flen], initializer=tf.constant_initializer(1.))
        trg_yhat, src_yhat = label_propagate(src_e, trg_e, T.src_y, FLAGS.bs, sigma, FLAGS.lpc, FLAGS.lp_iter)
        trg_test_yhat, _ = label_propagate(src_test_e, trg_test_e, T.src_test_y, FLAGS.bs, sigma, FLAGS.lpc,
                                           FLAGS.lp_iter)
        # loss_lp = T.adpt * lw * lp_loss(T.src_y, src_yhat)
        loss_lp = T.adpt * lw * lp_loss(T.src_y, src_yhat, FLAGS.lp_loss)
    else:
        loss_lp = constant(0)

    with arg_scope([lgan_loss], classifier=classifier):
        loss_src_reg = lgan_loss(src_x, T.src_fake) if FLAGS.sw > 0 else constant(0)
        loss_trg_reg = lgan_loss(trg_x, T.trg_fake) if FLAGS.tw > 0 else constant(0)

    # Weight decay
    if FLAGS.wd > 0:
        if FLAGS.dw > 0:
            var_d = tf.get_collection('trainable_variables', 'disc')
            d_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_d])
        else:
            d_decay = constant(0)
        # var_g = tf.get_collection('trainable_variables', 'class')
        var_g = tf.get_collection('trainable_variables', 'resnet_model') \
                + tf.get_collection('trainable_variables', 'class')
        g_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_g])
    else:
        d_decay = constant(0)
        g_decay = constant(0)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    # var_class = tf.get_collection('trainable_variables', 'class/')
    var_class = tf.get_collection('trainable_variables', 'resnet_model/') \
                + tf.get_collection('trainable_variables', 'class')
    ema_op = ema.apply(var_class)
    src_ema_p, _ = classifier(src_test_x, phase=False, getter=tb.tfutils.get_getter(ema))
    trg_ema_p, _ = classifier(trg_test_x, phase=False, getter=tb.tfutils.get_getter(ema))
    # src_test_p = nn.classifier(src_test_x, phase=False)
    # trg_test_p = nn.classifier(trg_test_x, phase=False)

    # Accuracies
    src_acc = accuracy(T.src_y, src_p)
    trg_acc = accuracy(T.trg_y, trg_p)
    src_test_acc = accuracy(T.src_test_y, src_test_p)
    trg_test_acc = accuracy(T.trg_test_y, trg_test_p)
    trg_test_lp_acc = accuracy(T.trg_test_y, trg_test_yhat) if FLAGS.lw > 0 else constant(0)
    fn_src_test_acc = tb.function(T.sess, [T.src_test_x, T.src_test_y], src_test_acc)
    fn_trg_test_acc = tb.function(T.sess, [T.trg_test_x, T.trg_test_y], trg_test_acc)
    fn_trg_test_lp_acc = tb.function(T.sess, [T.src_test_x, T.trg_test_x, T.src_test_y, T.trg_test_y],
                                     trg_test_lp_acc)
    src_ema_acc = accuracy(T.src_test_y, src_ema_p)
    trg_ema_acc = accuracy(T.trg_test_y, trg_ema_p)
    fn_src_ema_acc = tb.function(T.sess, [T.src_test_x, T.src_test_y], src_ema_acc)
    fn_trg_ema_acc = tb.function(T.sess, [T.trg_test_x, T.trg_test_y], trg_ema_acc)

    # Optimizer
    if FLAGS.dw > 0:
        loss_disc = (loss_disc +
                     wd * d_decay)
        var_disc = tf.get_collection('trainable_variables', 'disc')
        # train_disc = tf.train.AdamOptimizer(FLAGS.lr, 0.5).minimize(loss_disc, var_list=var_disc)
        train_disc = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9) \
            .minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    loss_main = (cw * loss_src_class +
                 T.adpt * dw * loss_dann +
                 # lw * loss_lp +
                 wd * g_decay +
                 xw * loss_trg_cent +
                 sw * loss_src_reg +
                 tw * loss_trg_reg
                 # T.adpt * sw * loss_src_reg +
                 # T.adpt * tw * loss_trg_reg
                 )

    # var_main = tf.get_collection('trainable_variables', 'class')
    var_main = tf.get_collection('trainable_variables', 'resnet_model') \
               + tf.get_collection('trainable_variables', 'class')
    # train_main = tf.train.AdamOptimizer(FLAGS.lr, 0.5).minimize(loss_main, var_list=var_main)
    # train_main = tf.group(train_main, ema_op)

    if FLAGS.lw:
        # trg_grad_w = get_grad_weight(trg_yhat, flen)
        trg_grad_w = get_grad_weight(trg_yhat, flen, FLAGS.grad_val)
        if not FLAGS.cgw: trg_grad_w = tf.ones_like(trg_grad_w)
        """
        # In the past: if FLAGS.cgw: trg_grad_w = tf.ones_like(trg_grad_w)
        """
        T.trg_grad_w = trg_grad_w

        for i in range(len(var_main)):
            if 'sigma' in var_main[i].name:
                sig_idx = i
            # for name_madi in var_main[i].name.split('/'):
            #     if 'sigma' in name_madi:
            #         sig_idx = i
            #         print(sig_idx)

        cyc_sf_grad = tf.gradients(loss_lp, src_e)
        cyc_tf_grad = tf.gradients(loss_lp, trg_e)
        sf_grad_w = tf.gradients(src_e, var_main, cyc_sf_grad)
        cyc_sigma_grad = tf.gradients(loss_lp, var_main[sig_idx])

        # to evaluate the effect of weighted grad
        weighted_grad = [cyc_tf_grad[0][:FLAGS.bs] * trg_grad_w]
        tf_grad_w = tf.gradients(trg_e, var_main, weighted_grad)

    # else:
    # sig_idx = None
    # sf_grad_w = None
    # tf_grad_w = None
    # cyc_sigma_grad = None
    step_wo_disc, _ = get_decay_var_op(name='wo_inc')

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    grad = []

    wo_smooth_grad_w = tf.gradients(loss_main, var_main)
    # cyc_sf_grad = tf.gradients(loss_lp, src_e)
    # cyc_tf_grad = tf.gradients(loss_lp, trg_e)
    # sf_grad_w = tf.gradients(src_e, var_main, cyc_sf_grad)
    # cyc_sigma_grad = tf.gradients(loss_lp, var_main[sig_idx])
    #
    # # to evaluate the effect of weighted grad
    # weighted_grad = [cyc_tf_grad[0][:FLAGS.bs] * trg_grad_w]
    # tf_grad_w = tf.gradients(trg_e, var_main, weighted_grad)

    for i in range(len(var_main)):
        if FLAGS.lw == 0:
            grad.append(wo_smooth_grad_w[i])
        else:
            if i == sig_idx:
                grad.append(wo_smooth_grad_w[i] + cyc_sigma_grad[0])
            else:
                if sf_grad_w[i] == None and tf_grad_w[i] == None:
                    grad.append(wo_smooth_grad_w[i])
                    # print("both none")
                elif sf_grad_w[i] == None:
                    grad.append(wo_smooth_grad_w[i] + tf_grad_w[i])
                    # print("sf none")
                elif tf_grad_w[i] == None:
                    grad.append(wo_smooth_grad_w[i] + sf_grad_w[i])
                    # print("tf none")
                else:
                    grad.append(wo_smooth_grad_w[i] + sf_grad_w[i] + tf_grad_w[i])
                    # print("both not none")
    # train_main = tf.train.AdamOptimizer(FLAGS.lr, 0.5).apply_gradients(list(zip(grad, var_main)),
    #                                                                    global_step=step_wo_disc)
    train_main = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9) \
        .apply_gradients(list(zip(grad, var_main)), global_step=step_wo_disc)
    train_main = tf.group(train_main, ema_op)

    # Summarizations
    summary_disc = [tf.summary.scalar('domain/loss_disc', loss_disc), ]
    summary_main = [tf.summary.scalar('domain/loss_dann', loss_dann),
                    tf.summary.scalar('class/loss_src_class', loss_src_class),
                    tf.summary.scalar('lp/loss_lp', loss_lp),
                    tf.summary.scalar('smooth/loss_trg_cent', loss_trg_cent),
                    tf.summary.scalar('smooth/loss_src_reg', loss_src_reg),
                    tf.summary.scalar('smooth/loss_trg_reg', loss_trg_reg),
                    tf.summary.scalar('hyper/cw', cw),
                    tf.summary.scalar('hyper/dw', dw),
                    tf.summary.scalar('hyper/lw', lw),
                    tf.summary.scalar('hyper/xw', xw),
                    tf.summary.scalar('hyper/sw', sw),
                    tf.summary.scalar('hyper/tw', tw),
                    tf.summary.scalar('acc/src_acc', src_acc),
                    tf.summary.scalar('acc/trg_acc', trg_acc)]

    # Merge summaries
    summary_disc = tf.summary.merge(summary_disc)
    summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('class'), loss_src_class,
                   c('disc'), loss_disc,
                   c('dann'), loss_dann,
                   c('lp'), loss_lp,
                   c('cent'), loss_trg_cent,
                   c('src_reg'), loss_src_reg,
                   c('trg_reg'), loss_trg_reg,
                   c('src_acc'), src_acc,
                   c('trg_acc'), trg_acc
                   ]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_src_test_acc = fn_src_test_acc
    T.fn_trg_test_acc = fn_trg_test_acc
    T.fn_trg_test_lp_acc = fn_trg_test_lp_acc
    T.fn_src_ema_acc = fn_src_ema_acc
    T.fn_trg_ema_acc = fn_trg_ema_acc

    print("============================LPDA model initialization ended.============================")

    return T
