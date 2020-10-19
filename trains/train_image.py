"""
Parts of codes are from
https://github.com/RuiShu/dirt-t/codebase/train.py
"""

import datetime
import os
import time
from collections import deque
from statistics import mean
import csv

import numpy as np
import tensorbayes as tb
import tensorflow as tf
from pytz import timezone
from termcolor import colored

from data.dataset import get_data, get_info
from utils import delete_existing, save_value, save_model, adaptation_factor, normalize


def update_dict(M, feed_dict, FLAGS, src=None, trg=None, Ls=None, Lt=None):
    if src:
        src_x, src_y = src.train.next_batch(FLAGS.bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})
        if Ls:
            z = np.random.normal(0, FLAGS.sv, (FLAGS.bs, FLAGS.src_nz))
            if FLAGS.pn:
                z = normalize(z)
                z = FLAGS.sv * z
            src_fake = Ls.sess.run(Ls.fake_x, feed_dict={Ls.x: src_x, Ls.z: z})
            feed_dict.update({M.src_fake: src_fake})
    if trg:
        trg_x, trg_y = trg.train.next_batch(FLAGS.bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})
        if Lt:
            z = np.random.normal(0, FLAGS.tv, (FLAGS.bs, FLAGS.trg_nz))
            if FLAGS.pn:
                z = normalize(z)
                z = FLAGS.tv * z
            trg_fake = Lt.sess.run(Lt.fake_x, feed_dict={Lt.x: trg_x, Lt.z: z})
            feed_dict.update({M.trg_fake: trg_fake})


def stopping_criteria(past_acc, current_acc):
    if current_acc < 0.5:
        print(f"Trg_test_ema, {round(current_acc, 5)} is less than 0.5")
    if current_acc < past_acc:
        print(f"Trg_test_ema, {round(current_acc, 5)} is less than last save point trg_test_ema, {round(past_acc, 5)}")
    return (current_acc < 0.5) | (current_acc < past_acc)


def train_image(M, FLAGS, Ls=None, Lt=None, saver=None, model_name=None):
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
    max_trg_test_ema = 0.0
    past_trg_test_ema = 0.0

    if FLAGS.adpt:
        adpt = adaptation_factor(0, FLAGS.adpt_val)
    else:
        adpt = 1.
    feed_dict = {M.adpt: adpt}

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
    start_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %I:%M:%S %p')
    print(colored(start_time, "blue"))
    epoch_time = deque(maxlen=5)
    temp_time = time.time()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops = [v for v in update_ops if 'lgan/' not in v.name]

    for i in range(n_epoch * iterep):
        # Train the discriminator
        # For batch normalization

        update_dict(M, feed_dict, FLAGS, src, trg, Ls, Lt)
        # summary, _ = M.sess.run(M.ops_disc, feed_dict)
        summary = M.sess.run([M.ops_disc, update_ops], feed_dict)[0][0]
        train_writer.add_summary(summary, i + 1)

        # Train the generator and the classifier
        update_dict(M, feed_dict, FLAGS, src, trg, Ls, Lt)
        # summary, _ = M.sess.run(M.ops_main, feed_dict)
        summary = M.sess.run([M.ops_main, update_ops], feed_dict)[0][0]
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        ########################################
        # train_src_x = M.sess.run(M.temp_src_x, feed_dict)
        # train_trg_x = M.sess.run(M.temp_trg_x, feed_dict)
        # test_src_x = src.test.images
        # test_src_y = src.test.labels
        # idx = np.random.permutation(range(len(test_src_x)))
        # test_src_x = test_src_x[idx]
        # test_src_y = test_src_y[idx]
        # test_trg_x = trg.test.images
        # test_trg_y = trg.test.labels
        # idx = np.random.permutation(range(len(test_trg_x)))
        # test_trg_x = test_trg_x[idx]
        # test_trg_y = test_trg_y[idx]
        # import matplotlib.pyplot as plt
        # for j in range(5):
        #     print(feed_dict[M.src_y][j])
        #     print(np.max(train_src_x[j]))
        #     print(np.min(train_src_x[j]))
        #     plt.imshow(train_src_x[j])
        #     plt.show()
        #     print(feed_dict[M.trg_y][j])
        #     print(np.max(train_trg_x[j]))
        #     print(np.min(train_trg_x[j]))
        #     plt.imshow(train_trg_x[j])
        #     plt.show()
        #     print(test_src_y[j])
        #     print(np.max(test_src_x[j]))
        #     print(np.min(test_src_x[j]))
        #     plt.imshow(test_src_x[j])
        #     plt.show()
        #     print(test_trg_y[j])
        #     print(np.max(test_trg_x[j]))
        #     print(np.min(test_trg_x[j]))
        #     plt.imshow(test_trg_x[j])
        #     plt.show()
        #
        # import sys
        # sys.exit()
        ########################################

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(i, n_epoch * iterep),
                                            display=True)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            for j, item in enumerate(print_list):
                if j % 2 == 0:
                    print_list[j] = item.decode("ascii")
                else:
                    print_list[j] = round(item, 5)

            print(print_list)
            print_list = []

            save_value(M.fn_src_test_acc, 'test/src_test',
                       src.test, train_writer, i + 1, print_list, bs=FLAGS.bs)
            save_value(M.fn_src_ema_acc, 'test/src_test_ema',
                       src.test, train_writer, i + 1, print_list, bs=FLAGS.bs)
            save_value(M.fn_trg_test_acc, 'test/trg_test',
                       trg.test, train_writer, i + 1, print_list, bs=FLAGS.bs)
            # save_value(M.fn_trg_test_lp_acc, 'test/trg_test_lp',
            #            [src.test, trg.test], train_writer, i + 1, print_list, lp=True)
            trg_test_ema = save_value(M.fn_trg_ema_acc, 'test/trg_test_ema',
                                      trg.test, train_writer, i + 1, print_list, bs=FLAGS.bs)

            print_list += ['epoch', epoch]
            print(print_list)

            if max_trg_test_ema < trg_test_ema:
                max_trg_test_ema = trg_test_ema

            if FLAGS.adpt:
                adpt = adaptation_factor(epoch / n_epoch, FLAGS.adpt_val)
            else:
                adpt = 1.
            feed_dict.update({M.adpt: adpt})

            epoch_time.appendleft(time.time() - temp_time)
            temp_time = time.time()
            needed_time = mean(epoch_time) * (n_epoch - epoch)
            eta = datetime.datetime.now(timezone('Asia/Seoul')) + datetime.timedelta(seconds=needed_time)
            print(f"Needed time: {str(datetime.timedelta(seconds=round(needed_time)))}, "
                  f"ETA: {eta.strftime('%Y-%m-%d %I:%M:%S %p')}, Max_trg_test_ema: "
                  + colored(round(max_trg_test_ema, 5), "red"))
        if saver and (i + 1) % itersave == 0:
            save_model(saver, M, model_dir, i + 1)
            if stopping_criteria(past_trg_test_ema, trg_test_ema):
                break
            past_trg_test_ema = trg_test_ema

    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)

    print(f"Max_trg_test_ema: {max_trg_test_ema}")
    print("============================LPDA training ended.============================")
    end_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %I:%M:%S %p')
    print(colored(end_time, "blue"))

    # with open("result.txt", "a") as result_file:
    #     result_file.write(model_name + "\n")
    #     result_file.write(str(max_trg_test_ema) + "\n")
    #     result_file.close()

    with open('result.csv', 'a') as f:
        csv_writer = csv.writer(f)
        write_list = [model_name, max_trg_test_ema]
        csv_writer.writerow(write_list)
