import os

import numpy as np
import tensorbayes as tb
import tensorflow as tf
from statistics import mean
from collections import deque
from pytz import timezone
import datetime
from termcolor import colored
import time

from data.dataset import get_data, get_info
from utils import delete_existing, save_model


def update_dict(L, feed_dict, FLAGS, data=None):
    if data:
        data_x, _ = data.train.next_batch(FLAGS.bs)
        feed_dict.update({L.x: data_x})
    z = np.random.normal(0, FLAGS.lgan_var, (FLAGS.bs, FLAGS.nz))
    pos = np.eye(FLAGS.nz)[np.random.permutation(FLAGS.nz)[:FLAGS.jcb]]
    pos = np.tile(np.expand_dims(pos, 0), [FLAGS.bs, 1, 1])
    pos = np.reshape(np.transpose(pos, [1, 0, 2]), [FLAGS.bs * FLAGS.jcb, -1])
    iorth = np.tile(np.expand_dims(np.eye(FLAGS.jcb), 0), [FLAGS.bs, 1, 1])
    feed_dict.update({L.z: z, L.pos: pos, L.iorth: iorth})


def lgan_train(L, FLAGS, saver=None, lgan_name=None):
    """
    :param L: (TensorDict) the model
    :param FLAGS: (FLAGS) contains experiment info
    :param saver: (Saver) saves models during training
    :param lgan_name: name of the lgan model being run with relevant parms info
    :return: None
    """
    bs = FLAGS.bs
    lrD = FLAGS.lrD
    lrG = FLAGS.lrG
    iterep = 1000
    itersave = 20000
    n_epoch = 160
    epoch = 0
    feed_dict = {L.lrD: lrD, L.lrG: lrG}

    # Create a log directory and FileWriter
    log_dir = os.path.join(FLAGS.logdir, lgan_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create a save directory
    if saver:
        model_dir = os.path.join(FLAGS.ckptdir, lgan_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    data = get_data(FLAGS.data, FLAGS)
    get_info(data, FLAGS.bs)

    print(f"Batch size: {bs}")
    print(f"Iterep: {iterep}")
    print(f"Total iterations: {n_epoch * iterep}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {model_dir}")

    print("============================LGAN training started.============================")
    start_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %I:%M:%S %p')
    print(colored(start_time, "blue"))
    epoch_time = deque(maxlen=5)
    temp_time = time.time()

    for i in range(n_epoch * iterep):
        # Train the discriminator
        update_dict(L, feed_dict, FLAGS, data)
        summary, _ = L.sess.run(L.ops_disc, feed_dict)
        train_writer.add_summary(summary, i + 1)

        # Train the generator and the classifier
        update_dict(L, feed_dict, FLAGS, data)
        summary, _ = L.sess.run(L.ops_gen, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(i, n_epoch * iterep),
                                            display=True)

        if end_epoch:
            summary = L.sess.run(L.ops_image, feed_dict)
            train_writer.add_summary(summary, i + 1)
            train_writer.flush()

            lrD *= FLAGS.lrDecay
            lrG *= FLAGS.lrDecay
            feed_dict.update({L.lrD: lrD, L.lrG: lrG})
            print_list = L.sess.run(L.ops_print, feed_dict)

            for j, item in enumerate(print_list):
                if j % 2 == 0:
                    print_list[j] = item.decode("ascii")
                else:
                    print_list[j] = round(item, 5)

            print_list += ['epoch', epoch]
            print(print_list)

            epoch_time.appendleft(time.time() - temp_time)
            temp_time = time.time()
            needed_time = mean(epoch_time) * (n_epoch - epoch)
            eta = datetime.datetime.now(timezone('Asia/Seoul')) + datetime.timedelta(seconds=needed_time)
            print(f"Needed time: {str(datetime.timedelta(seconds=round(needed_time)))}, "
                  f"ETA: {eta.strftime('%Y-%m-%d %I:%M:%S %p')}")

        if saver and (i + 1) % itersave == 0:
            save_model(saver, L, model_dir, i + 1)

    # Saving final model
    if saver:
        save_model(saver, L, model_dir, i + 1)
    print("============================LGAN training ended.============================")
    end_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %I:%M:%S %p')
    print(colored(end_time, "blue"))
