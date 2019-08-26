import tensorflow as tf
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorflow.contrib.framework import add_arg_scope

eps = 1e-8


def lp_loss(y, yhat, scope=None):
    with tf.name_scope(scope, 'lgan_loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y - yhat), 1))
    return loss


@add_arg_scope
def lgan_loss(real_x, fake_x, classifier, scope=None):
    with tf.name_scope(scope, 'lgan_loss'):
        real_p = classifier(real_x, phase=True)
        fake_p = classifier(fake_x, phase=True)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(real_p), logits=fake_p))
    return loss


def metric_mat(src_e=None, trg_e=None, bs=0, sigma=0):
    w_ss = w_st = w_ts = w_tt = None
    sigma_tile = tf.tile(tf.expand_dims(tf.expand_dims(sigma, 1), 2), [1, bs, bs])
    if src_e != None:
        src_fb1 = tf.expand_dims(tf.transpose(src_e, [1, 0]), 2)
        src_f1b = tf.transpose(src_fb1, [0, 2, 1])
        ## SS ##
        # print('Get src_f, src_fb1, src_f1b shape')
        # print(src_e.get_shape())
        # print(src_fb1.get_shape())
        # print(src_f1b.get_shape())
        m_ss = tf.square(src_fb1 - src_f1b) + eps
        m_ss = tf.reduce_mean(m_ss * tf.square(sigma_tile), 0)
        w_ss = tf.exp(-m_ss / 2)
        w_ss = w_ss - tf.diag(tf.diag_part(w_ss))
    if trg_e != None:
        trg_fb1 = tf.expand_dims(tf.transpose(trg_e, [1, 0]), 2)
        trg_f1b = tf.transpose(trg_fb1, [0, 2, 1])
        ## TT ##
        m_tt = tf.square(trg_fb1 - trg_f1b) + eps
        m_tt = tf.reduce_mean(m_tt * tf.square(sigma_tile), 0)
        w_tt = tf.exp(-m_tt / 2)
        w_tt = w_tt - tf.diag(tf.diag_part(w_tt))
    if src_e != None and trg_e != None:
        ## ST & TS ##
        m_st = tf.square(src_fb1 - trg_f1b)
        m_st = tf.reduce_mean(m_st * tf.square(sigma_tile), 0)
        w_st = tf.exp(-m_st / 2)
        w_ts = tf.transpose(w_st, [1, 0])

    return w_ss, w_st, w_ts, w_tt


def lp_closed(trans_list, l):
    a_ul, a_uu = trans_list
    a_ul = a_ul + eps
    a_uu = a_uu + eps
    node_num_u, node_num_l = a_ul.get_shape().as_list()
    # if node_num_u == None:
    #     node_num = node_num_l
    # else:
    #     node_num = node_num_u
    if node_num_l == None:
        node_num = node_num_u
    else:
        node_num = node_num_l

    a = tf.concat([a_ul, a_uu], 1)
    d = tf.tile(tf.reduce_sum(a, 1, keepdims=True), [1, node_num])
    p_ul = a_ul / d
    p_uu = a_uu / d
    return tf.matmul(tf.matrix_inverse(tf.eye(node_num) - p_uu), tf.matmul(p_ul, l))


def lp_iter(trans_list, l, u):
    a_ul, a_uu = trans_list
    a_ul = a_ul + eps
    a_uu = a_uu + eps
    node_num_u, node_num_l = a_ul.get_shape().as_list()
    if node_num_u == None:
        node_num = node_num_l
    else:
        node_num = node_num_u
    a = tf.concat([a_ul, a_uu], 1)
    d = tf.tile(tf.reduce_sum(a, 1, keepdims=True), [1, node_num])
    p_ul = a_ul / d
    p_uu = a_uu / d
    return tf.matmul(p_ul, l) + tf.matmul(p_uu, u)


def label_propagate(src_e, trg_e, src_y, bs, sigma, lpc, iter, scope=None):
    w_ss, w_st, w_ts, w_tt = metric_mat(src_e, trg_e, bs, sigma)
    if lpc:
        trg_yhat = lp_closed([w_ts, w_tt], src_y)
        src_yhat = lp_closed([w_st, w_ss], trg_yhat)
    else:
        trg_yhat = tf.ones_like(src_y)
        print(trg_yhat.get_shape())
        trg_yhat = trg_yhat / tf.reduce_sum(trg_yhat, 1, keepdims=True)
        src_yhat = tf.identity(trg_yhat)
        for _ in range(iter):
            trg_yhat = lp_iter([w_ts, w_tt], src_y, trg_yhat)
        for _ in range(iter):
            src_yhat = lp_iter([w_st, w_ss], trg_yhat, src_yhat)

    return trg_yhat, src_yhat
