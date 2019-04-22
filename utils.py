"""
Parts of codes are from
https://github.com/RuiShu/dirt-t/codebase/models/extra_layers.py
https://github.com/RuiShu/dirt-t/codebase/utils.py
"""

import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorbayes as tb
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope


def preprocessing(inputs, exp_sz, exp_ch):
    # The inputs should be in NHWC format
    inputs = tf.cast(inputs, tf.float32)
    if int(inputs.get_shape()[1] == inputs.get_shape()[2]):
        sz = int(inputs.get_shape()[1])
    else:
        raise ValueError("For the preprocessing, the inputs should be formatted in NHWC.")
    ch = int(inputs.get_shape()[-1])
    if ch == 1 and exp_ch == 3:
        logging.info("Converting gray-scale images to rgb-scale")
        inputs = tf.image.grayscale_to_rgb(inputs)
    elif ch == 3 and exp_ch == 1:
        logging.info("Converting rgb-scale images to gray-scale")
        inputs = tf.image.rgb_to_grayscale(inputs)
    if sz != exp_sz:
        logging.info(f"Resizing images from {sz} to {exp_sz}")
        inputs = tf.image.resize_images(inputs, [exp_sz, exp_sz])

    return inputs


def normalize(x):
    square_sum = np.sum(np.square(x), axis=tuple(range(1, len(x.shape))), keepdims=True)
    x_inv_norm = 1. / np.sqrt(np.maximum(square_sum, 1e-12))
    return np.multiply(x, x_inv_norm)


def print_image(image, iter):
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 1, 3))

    for i in range(iter):
        plt.imshow(image[i])
        plt.show()


def delete_existing(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_model(saver, M, model_dir, global_step):
    path = saver.save(M.sess, os.path.join(model_dir, 'model'), global_step=global_step)
    print(f"Saving model to {path}")


def save_value(fn_val, tag, data,
               train_writer=None, global_step=None, print_list=None,
               full=True):
    """Log fn_val evaluation to tf.summary.FileWriter
    fn_val       - (fn) Takes (x, y) as input and returns value
    tag          - (str) summary tag for FileWriter
    data         - (Data) data object with images/labels attributes
    train_writer - (FileWriter)
    global_step  - (int) global step in file writer
    print_list   - (list) list of vals to print to stdout
    full         - (bool) use full dataset v. first 1000 samples
    """
    acc, summary = compute_value(fn_val, tag, data, full)
    train_writer.add_summary(summary, global_step)
    acc = round(acc, 3)
    print_list += [os.path.basename(tag), acc]


def compute_value(fn_val, tag, data, full=True):
    """Compute value w.r.t. data
    fn_val - (fn) Takes (x, y) as input and returns value
    tag    - (str) summary tag for FileWriter
    data   - (Data) data object with images/labels attributes
    full   - (bool) use full dataset v. first 1024 samples
    """
    with tb.nputils.FixedSeed(0):
        shuffle = np.random.permutation(len(data.images))

    xs = data.images[shuffle]
    ys = data.labels[shuffle] if data.labels is not None else None

    if not full:
        xs = xs[:1024]
        ys = ys[:1024] if ys is not None else None

    acc = 0.
    n = len(xs)
    bs = 128

    for i in range(0, n, bs):
        # x = data.preprocess(xs[i:i + bs])
        x = xs[i:i + bs]
        y = ys[i:i + bs]
        acc += fn_val(x, y) / n * len(x)

    summary = tf.Summary.Value(tag=tag, simple_value=acc)
    summary = tf.Summary(value=[summary])
    return acc, summary


@add_arg_scope
def accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output


@add_arg_scope
def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output


@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)


@add_arg_scope
def relu(x, name=None):
    return tf.nn.relu(x, name=name)


@add_arg_scope
def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)


@add_arg_scope
def sigmoid(x, name=None):
    return tf.nn.sigmoid(x, name=name)


@add_arg_scope
def reshape(x, shape, name=None):
    return tf.reshape(x, shape, name)


@add_arg_scope
def global_pool(x, axis, keepdims=False, name=None):
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims, name=name)
