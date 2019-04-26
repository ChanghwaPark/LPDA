"""
Parts of codes are from
https://github.com/RuiShu/dirt-t/codebase/models/dirtt.py
"""

import importlib

import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder, constant
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent

from data.dataset import get_attr
from utils import accuracy, preprocessing, get_decay_var_op, get_grad_weight
from .loss import lp_loss, label_propagate


def lpda(FLAGS):
    """
    :param FLAGS: Contains the experiment info
    :return: (TensorDict) the model
    """

    print("============================LPDA model initialization started.============================")

    nn_path = importlib.import_module(".{}".format(FLAGS.nn), package='networks')
    nn_class = getattr(nn_path, FLAGS.nn)
    nn = nn_class(FLAGS)
    src_sz, trg_sz, exp_sz, src_ch, trg_ch, exp_ch, nc = get_attr(FLAGS.src, FLAGS.trg)
    print(f"Image size: {exp_sz}, number of channels: {exp_ch}, number of classes: {nc}")

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        src_x=placeholder((None, src_sz, src_sz, src_ch)),
        src_y=placeholder((None, nc)),
        trg_x=placeholder((None, trg_sz, trg_sz, trg_ch)),
        trg_y=placeholder((None, nc)),
        src_test_x=placeholder((None, src_sz, src_sz, src_ch)),
        trg_test_x=placeholder((None, trg_sz, trg_sz, trg_ch)),
        test_y=placeholder((None, nc)),
        adpt=placeholder(None)
    ))

    cw = constant(1)
    dw = constant(FLAGS.dw)
    lw = constant(FLAGS.lw)
    wd = constant(FLAGS.wd)

    # Preprocess images to experiment size and channels
    src_x = preprocessing(T.src_x, exp_sz, exp_ch)
    trg_x = preprocessing(T.trg_x, exp_sz, exp_ch)
    src_test_x = preprocessing(T.src_test_x, exp_sz, exp_ch)
    trg_test_x = preprocessing(T.trg_test_x, exp_sz, exp_ch)

    # The features and the predictions of the source and the target images
    src_e = nn.classifier(src_x, phase=True, enc_phase=True, trim=FLAGS.trim)
    trg_e = nn.classifier(trg_x, phase=True, enc_phase=True, trim=FLAGS.trim, internal_update=True)

    src_p = nn.classifier(src_e, phase=True, enc_phase=False, trim=FLAGS.trim)
    trg_p = nn.classifier(trg_e, phase=True, enc_phase=False, trim=FLAGS.trim, internal_update=True)

    flen = nn.get_flen()

    # Source true label cross entropy minimization
    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))

    # Adversarial domain confusion
    if FLAGS.dw > 0:
        src_logit = nn.feature_discriminator(src_e, phase=True)
        trg_logit = nn.feature_discriminator(trg_e, phase=True)

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
        loss_lp = T.adpt * lw * lp_loss(T.src_y, src_yhat)
    else:
        loss_lp = constant(0)

    # Weight decay
    if FLAGS.wd > 0:
        if FLAGS.dw > 0:
            var_d = tf.get_collection('trainable_variables', 'disc')
            d_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_d])
        else:
            d_decay = constant(0)
        var_g = tf.get_collection('trainable_variables', 'class')
        g_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_g])
    else:
        d_decay = constant(0)
        g_decay = constant(0)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    src_ema_p = nn.classifier(src_test_x, phase=False, getter=tb.tfutils.get_getter(ema))
    trg_ema_p = nn.classifier(trg_test_x, phase=False, getter=tb.tfutils.get_getter(ema))
    src_test_p = nn.classifier(src_test_x, phase=False)
    trg_test_p = nn.classifier(trg_test_x, phase=False)

    # Accuracies
    src_acc = accuracy(T.src_y, src_p)
    trg_acc = accuracy(T.trg_y, trg_p)
    src_test_acc = accuracy(T.test_y, src_test_p)
    trg_test_acc = accuracy(T.test_y, trg_test_p)
    fn_src_test_acc = tb.function(T.sess, [T.src_test_x, T.test_y], src_test_acc)
    fn_trg_test_acc = tb.function(T.sess, [T.trg_test_x, T.test_y], trg_test_acc)
    src_ema_acc = accuracy(T.test_y, src_ema_p)
    trg_ema_acc = accuracy(T.test_y, trg_ema_p)
    fn_src_ema_acc = tb.function(T.sess, [T.src_test_x, T.test_y], src_ema_acc)
    fn_trg_ema_acc = tb.function(T.sess, [T.trg_test_x, T.test_y], trg_ema_acc)

    # Optimizer
    if FLAGS.dw > 0:
        loss_disc = (loss_disc +
                     wd * d_decay
                     )
        var_disc = tf.get_collection('trainable_variables', 'disc')
        train_disc = tf.train.AdamOptimizer(FLAGS.lr, 0.5).minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    loss_main = (cw * loss_src_class +
                 T.adpt * dw * loss_dann +
                 # lw * loss_lp +
                 wd * g_decay
                 )

    var_main = tf.get_collection('trainable_variables', 'class')
    # train_main = tf.train.AdamOptimizer(FLAGS.lr, 0.5).minimize(loss_main, var_list=var_main)
    # train_main = tf.group(train_main, ema_op)

    if FLAGS.lw:
        trg_grad_w = get_grad_weight(trg_yhat, flen)
        if FLAGS.cgw: trg_grad_w = tf.ones_like(trg_grad_w)

        for i in range(len(var_main)):
            for name_madi in var_main[i].name.split('/'):
                if 'sigma' in name_madi:
                    sig_idx = i

    step_wo_disc, _ = get_decay_var_op(name='wo_inc')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grad = []

        wo_smooth_grad_w = tf.gradients(loss_main, var_main)
        cyc_sf_grad = tf.gradients(loss_lp, src_e)
        cyc_tf_grad = tf.gradients(loss_lp, trg_e)
        sf_grad_w = tf.gradients(src_e, var_main, cyc_sf_grad)
        cyc_sigma_grad = tf.gradients(loss_lp, var_main[sig_idx])

        # to evaluate the effect of weighted grad
        weighted_grad = cyc_tf_grad[0][:FLAGS.bs] * trg_grad_w
        tf_grad_w = tf.gradients(trg_e, var_main, weighted_grad)
        for i in range(len(var_main)):
            if i == sig_idx:
                # print('sigma grad')
                # print(var_main[i].name)
                grad.append(wo_smooth_grad_w[i] + cyc_sigma_grad[0])
            else:
                if sf_grad_w[i] == None and tf_grad_w[i] == None:
                    # print('sf none tf none')
                    # print(var_main[i].name)
                    grad.append(wo_smooth_grad_w[i])
                elif sf_grad_w[i] == None:
                    # print('sf none')
                    # print(var_main[i].name)
                    grad.append(wo_smooth_grad_w[i] + tf_grad_w[i])
                elif tf_grad_w[i] == None:
                    # print('tf none')
                    # print(var_main[i].name)
                    grad.append(wo_smooth_grad_w[i] + sf_grad_w[i])
                else:
                    # print('no none')
                    # print(var_main[i].name)
                    grad.append(wo_smooth_grad_w[i] + sf_grad_w[i] + tf_grad_w[i])
        train_main = tf.train.AdamOptimizer(FLAGS.lr, 0.5).apply_gradients(list(zip(grad, var_main)),
                                                                           global_step=step_wo_disc)
    train_main = tf.group(train_main, ema_op)

    # Summarizations
    summary_disc = [tf.summary.scalar('domain/loss_disc', loss_disc), ]
    summary_main = [tf.summary.scalar('domain/loss_dann', loss_dann),
                    tf.summary.scalar('class/loss_src_class', loss_src_class),
                    tf.summary.scalar('lp/loss_lp', loss_lp),
                    tf.summary.scalar('hyper/cw', cw),
                    tf.summary.scalar('hyper/dw', dw),
                    tf.summary.scalar('hyper/lw', lw),
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
                   c('lp'), loss_lp]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_src_test_acc = fn_src_test_acc
    T.fn_trg_test_acc = fn_trg_test_acc
    T.fn_src_ema_acc = fn_src_ema_acc
    T.fn_trg_ema_acc = fn_trg_ema_acc

    print("============================LPDA model initialization ended.============================")

    return T
