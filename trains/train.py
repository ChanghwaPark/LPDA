"""
Parts of codes are from
https://github.com/RuiShu/dirt-t/codebase/train.py
"""

import os

import numpy as np
import tensorbayes as tb
import tensorflow as tf

from data.dataset import get_data, get_info
from utils import delete_existing, save_value, save_model, print_image, normalize


def update_dict(M, feed_dict, FLAGS, src=None, trg=None):
    if src:
        src_x, src_y = src.train.next_batch(FLAGS.bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})
    if trg:
        trg_x, trg_y = trg.train.next_batch(FLAGS.bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})


def train(M, FLAGS, saver=None, model_name=None):
    """
    :param L: (TensorDIct) the LGAN model
    :param M: (TensorDict) the model
    :param FLAGS: (FLAGS) contains experiment info
    :param saver: (Saver) saves models during training
    :param model_name: name of the model being run with relevant parms info
    :return: None
    """
    iterep = 1000
    itersave = 20000
    n_epoch = FLAGS.epoch
    epoch = 0
    feed_dict = {}

    # Create a log directory and FileWriter
    log_dir = os.path.join(FLAGS.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create a save directory
    if saver:
        model_dir = os.path.join(FLAGS.ckptdir, model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    src = get_data(FLAGS.src, FLAGS)
    get_info(src, FLAGS.bs)
    trg = get_data(FLAGS.trg, FLAGS)
    get_info(trg, FLAGS.bs)

    print(f"Batch size: {FLAGS.bs}")
    print(f"Iterep: {iterep}")
    print(f"Total iterations: {n_epoch * iterep}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {model_dir}")

    print("============================LPDA training started.============================")

    for i in range(n_epoch * iterep):
        # Train the discriminator
        update_dict(M, feed_dict, FLAGS, src, trg)
        summary, _ = M.sess.run(M.ops_disc, feed_dict)
        train_writer.add_summary(summary, i + 1)

        # Train the generator and the classifier
        update_dict(M, feed_dict, FLAGS, src, trg)
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=True)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            for j, item in enumerate(print_list):
                if j % 2 == 0:
                    print_list[j] = item.decode("ascii")
                else:
                    print_list[j] = round(item, 5)

            save_value(M.fn_trg_ema_acc, 'test/trg_train_ema_1k',
                       trg.train, train_writer, i + 1, print_list, full=False)
            save_value(M.fn_src_test_acc, 'test/src_test',
                       src.test, train_writer, i + 1, print_list)
            save_value(M.fn_trg_test_acc, 'test/trg_test',
                       trg.test, train_writer, i + 1, print_list)
            save_value(M.fn_src_ema_acc, 'test/src_test_ema',
                       src.test, train_writer, i + 1, print_list)
            save_value(M.fn_trg_ema_acc, 'test/trg_test_ema',
                       trg.test, train_writer, i + 1, print_list)

            print_list += ['epoch', epoch]
            print(print_list)

        if saver and (i + 1) % itersave == 0:
            save_model(saver, M, model_dir, i + 1)

    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)

    print("============================LPDA training ended.============================")
