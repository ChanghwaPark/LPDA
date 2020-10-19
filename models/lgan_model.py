import importlib

import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder, constant
from tensorflow.python.ops.losses.losses_impl import absolute_difference as abs_diff
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent

from data.dataset import get_attr, train_image_process
from utils import preprocessing


def lgan(FLAGS, inference=False):
    """
    :param FLAGS: Contains the experiment info
    :return: (TensorDict) the model
    """

    print("============================LGAN model initialization started.============================")

    # nn_path = importlib.import_module(".{}".format(FLAGS.lgan_nn), package='networks')
    nn_path = importlib.import_module("networks.{}".format(FLAGS.lgan_nn))
    nn_class = getattr(nn_path, FLAGS.lgan_nn)
    nn = nn_class(FLAGS)
    data_sz, data_ch, nc = get_attr(FLAGS.data)
    exp_sz = FLAGS.sz
    exp_ch = FLAGS.ch
    bs = FLAGS.bs

    alpha = constant(FLAGS.alpha)
    beta = constant(FLAGS.beta)
    theta = constant(FLAGS.theta)
    delta = constant(FLAGS.delta)

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        x=placeholder((bs, data_sz, data_sz, data_ch)),
        z=placeholder((bs, FLAGS.nz)),
        pos=placeholder((bs * FLAGS.jcb, FLAGS.nz)),
        iorth=placeholder((bs, FLAGS.jcb, FLAGS.jcb)),
        lrD=placeholder(None),
        lrG=placeholder(None),
        enc_z=placeholder((bs, FLAGS.nz))
    ))

    # Preprocess x to the experiment size
    if FLAGS.data in ['amazon', 'dslr', 'webcam', 'c', 'i', 'p']:
        T.real_x = real_x = train_image_process(T.x, exp_sz)
    else:
        T.real_x = real_x = preprocessing(T.x, exp_sz, exp_ch)

    # Compute G(x, z) and G(x, 0)
    # T.fake_x = fake_x = nn.generator(real_x, T.z, phase=True)

    # Return the model when inferencing LGAN model
    if inference:
        # Compute G(x, z) and G(x, 0)
        # T.fake_x = nn.generator(real_x, T.z, phase=False)
        T.fake_x = nn.generator(real_x, T.z, phase=True)
        print("============================LGAN model initialization ended.============================")
        return T
    else:
        # Compute G(x, z) and G(x, 0)
        T.fake_x = fake_x = nn.generator(real_x, T.z, phase=True)

    fake_x0 = nn.generator(real_x, tf.zeros_like(T.z), phase=True)

    # Compute discriminator logits
    real_logit = nn.discriminator(real_x, phase=True)
    fake_logit = nn.discriminator(fake_x, phase=True)
    fake0_logit = nn.discriminator(fake_x0, phase=True)

    # Adversarial generator
    loss_disc = tf.reduce_mean(
        sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
        sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit) +
        theta * sigmoid_xent(labels=tf.zeros_like(fake0_logit), logits=fake0_logit))

    loss_fake = tf.reduce_mean(
        sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit) +
        theta * sigmoid_xent(labels=tf.ones_like(fake0_logit), logits=fake0_logit))

    # Locality
    loss_local = tf.reduce_mean(abs_diff(labels=real_x, predictions=fake_x0))

    # Orthogonality
    pos = T.pos * delta
    tiled_real_x = tf.tile(real_x, [FLAGS.jcb, 1, 1, 1])
    pos_fake_x = nn.generator(tiled_real_x, pos, phase=True)
    neg_fake_x = nn.generator(tiled_real_x, -pos, phase=True)

    jx = (pos_fake_x - neg_fake_x) / (2 * delta)
    jx = tf.reshape(jx, [bs, FLAGS.jcb, -1])
    jx_t = tf.transpose(jx, [0, 2, 1])
    loss_orth = tf.reduce_mean(abs_diff(tf.matmul(jx, jx_t), T.iorth))

    loss_gen = loss_fake + alpha * loss_local + beta * loss_orth

    # Optimizer
    var_disc = tf.get_collection('trainable_variables', FLAGS.data + '/lgan/dsc')
    train_disc = tf.train.AdamOptimizer(T.lrD, 0.5).minimize(loss_disc, var_list=var_disc)

    var_gen = tf.get_collection('trainable_variables', FLAGS.data + '/lgan/gen')
    train_gen = tf.train.AdamOptimizer(T.lrG, 0.5).minimize(loss_gen, var_list=var_gen)

    # Summarizations
    summary_disc = [tf.summary.scalar('disc/loss_disc', loss_disc)]
    summary_gen = [tf.summary.scalar('gen/loss_gen', loss_gen),
                   tf.summary.scalar('gen/loss_fake', loss_fake),
                   tf.summary.scalar('gen/loss_local', loss_local),
                   tf.summary.scalar('gen/loss_orth', loss_orth),
                   tf.summary.scalar('hyper/alpha', alpha),
                   tf.summary.scalar('hyper/beta', beta),
                   tf.summary.scalar('hyper/theta', theta),
                   tf.summary.scalar('hyper/delta', delta),
                   tf.summary.scalar('hyper/lrD', T.lrD),
                   tf.summary.scalar('hyper/lrG', T.lrG),
                   tf.summary.scalar('hyper/var', FLAGS.lgan_var)]
                   # tf.summary.image('image/real_x', real_x),
                   # tf.summary.image('image/fake_x0', fake_x0),
                   # tf.summary.image('image/fake_x', fake_x)]

    summary_image = [tf.summary.image('image/real_x', real_x),
                     tf.summary.image('image/fake_x0', fake_x0),
                     tf.summary.image('image/fake_x', fake_x)]

    # Merge summaries
    summary_disc = tf.summary.merge(summary_disc)
    summary_gen = tf.summary.merge(summary_gen)
    summary_image = tf.summary.merge(summary_image)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('gen'), loss_gen,
                   c('fake'), loss_fake,
                   c('local'), loss_local,
                   c('orth'), loss_orth]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_gen = [summary_gen, train_gen]
    T.ops_image = summary_image

    print("============================LGAN model initialization ended.============================")

    return T
