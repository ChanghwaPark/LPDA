import pickle

import cv2
import tensorflow as tf
from numpy import array

from .data_loaders import *


class Data(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.pointer = 0

    def normalize_sample(self, x):
        x = x - x.mean(axis=(1, 2, 3), keepdims=True)
        x = x / x.std(axis=(1, 2, 3), keepdims=True)
        return x

    def invert_randomly(self, x):
        randu = np.random.uniform(size=(len(x)), low=0., high=1.)
        randu = (np.less(randu, 0.5)).astype(np.float32)
        randu = np.expand_dims(np.expand_dims(np.expand_dims(randu, 1), 1), 1)
        x = np.abs(x - randu)
        return x

    def next_batch(self, bs, shuffle=True, normalize=False, invert_randomly=False):
        if shuffle:
            idx = np.random.choice(len(self.images), bs, replace=False)
        else:
            idx = range(self.pointer, self.pointer + bs)
            self.pointer += bs
            if self.pointer + bs > len(self.images):
                self.pointer = 0
        x = self.images[idx]
        if normalize:
            x = self.normalize_sample(x)
        if invert_randomly:
            x = self.invert_randomly(x)
        y = self.labels[idx]
        return x, y


class DataImage(object):
    def __init__(self, images, labels, sz, nn, training=True):
        self.sz = sz
        self.nn = nn
        self.training = training
        if training:
            self.images = images
        else:
            self.images = test_image_process(images, sz, nn)
        self.labels = labels
        self.pointer = 0

    def next_batch(self, bs, shuffle=True):
        if shuffle:
            idx = np.random.choice(len(self.images), bs, replace=False)
        else:
            idx = range(self.pointer, self.pointer + bs)
            self.pointer += bs
            if self.pointer + bs > len(self.images):
                self.pointer = 0
        x = self.images[idx]
        y = self.labels[idx]
        return x, y


class usps(object):
    def __init__(self, FLAGS):
        usps_data = load_usps(val=FLAGS.val, scale28=True, zero_centre=FLAGS.zc)
        train_x = usps_data.train_X
        test_x = usps_data.test_X
        val_x = usps_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[usps_data.train_y.reshape(-1)]
        test_y = np.eye(10)[usps_data.test_y.reshape(-1)]
        val_y = np.eye(10)[usps_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class usps1800(object):
    def __init__(self, FLAGS):
        usps_data = load_usps(val=FLAGS.val, scale28=True, zero_centre=FLAGS.zc)
        train_x = usps_data.train_X
        test_x = usps_data.test_X
        val_x = usps_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[usps_data.train_y.reshape(-1)]
        test_y = np.eye(10)[usps_data.test_y.reshape(-1)]
        val_y = np.eye(10)[usps_data.val_y.reshape(-1)]

        random_idx = np.random.permutation(len(train_x))[:1800]
        random_idx.sort()
        train_x = train_x[random_idx]
        train_y = train_y[random_idx]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class mnist(object):
    def __init__(self, FLAGS):
        mnist_data = load_mnist(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = mnist_data.train_X
        test_x = mnist_data.test_X
        val_x = mnist_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[mnist_data.train_y.reshape(-1)]
        test_y = np.eye(10)[mnist_data.test_y.reshape(-1)]
        val_y = np.eye(10)[mnist_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class mnist2000(object):
    def __init__(self, FLAGS):
        mnist_data = load_mnist(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = mnist_data.train_X
        test_x = mnist_data.test_X
        val_x = mnist_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[mnist_data.train_y.reshape(-1)]
        test_y = np.eye(10)[mnist_data.test_y.reshape(-1)]
        val_y = np.eye(10)[mnist_data.val_y.reshape(-1)]

        random_idx = np.random.permutation(len(train_x))[:2000]
        random_idx.sort()
        train_x = train_x[random_idx]
        train_y = train_y[random_idx]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class mnistm(object):
    def __init__(self, FLAGS):
        train, test, val = load_mnistm(os.path.join(FLAGS.datadir, 'mnistm'), val=FLAGS.val, zero_centre=FLAGS.zc)
        train_x = train[0]
        train_y = train[1]
        test_x = test[0]
        test_y = test[1]
        val_x = val[0]
        val_y = val[1]

        train_y = np.eye(10)[train_y.reshape(-1)]
        test_y = np.eye(10)[test_y.reshape(-1)]
        val_y = np.eye(10)[val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class svhn(object):
    def __init__(self, FLAGS):
        svhn_data = load_svhn(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = svhn_data.train_X
        test_x = svhn_data.test_X
        val_x = svhn_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[svhn_data.train_y.reshape(-1)]
        test_y = np.eye(10)[svhn_data.test_y.reshape(-1)]
        val_y = np.eye(10)[svhn_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class syndigits(object):
    def __init__(self, FLAGS):
        syndigits_data = load_syn_digits(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = syndigits_data.train_X
        test_x = syndigits_data.test_X
        val_x = syndigits_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[syndigits_data.train_y.reshape(-1)]
        test_y = np.eye(10)[syndigits_data.test_y.reshape(-1)]
        val_y = np.eye(10)[syndigits_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class cifar(object):
    def __init__(self, FLAGS):
        cifar_data = load_cifar10(val=FLAGS.val, range_01=FLAGS.zc)

        train_x = cifar_data.train_X
        test_x = cifar_data.test_X
        val_x = cifar_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(9)[cifar_data.train_y.reshape(-1)]
        test_y = np.eye(9)[cifar_data.test_y.reshape(-1)]
        val_y = np.eye(9)[cifar_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class stl(object):
    def __init__(self, FLAGS):
        stl_data = load_stl(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = stl_data.train_X
        test_x = stl_data.test_X
        val_x = stl_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(9)[stl_data.train_y.reshape(-1)]
        test_y = np.eye(9)[stl_data.test_y.reshape(-1)]
        val_y = np.eye(9)[stl_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class gtsrb(object):
    def __init__(self, FLAGS):
        gtsrb_data = load_gtsrb(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = gtsrb_data.train_X
        test_x = gtsrb_data.test_X
        val_x = gtsrb_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(43)[gtsrb_data.train_y.reshape(-1)]
        test_y = np.eye(43)[gtsrb_data.test_y.reshape(-1)]
        val_y = np.eye(43)[gtsrb_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class synsigns(object):
    def __init__(self, FLAGS):
        synsigns_data = load_syn_signs(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = synsigns_data.train_X
        test_x = synsigns_data.test_X
        val_x = synsigns_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(43)[synsigns_data.train_y.reshape(-1)]
        test_y = np.eye(43)[synsigns_data.test_y.reshape(-1)]
        val_y = np.eye(43)[synsigns_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class amazon(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'office/amazon_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(31)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class dslr(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'office/dslr_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(31)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class webcam(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'office/webcam_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(31)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class c(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'image-clef/c_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(12)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class i(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'image-clef/i_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(12)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class p(object):
    def __init__(self, FLAGS):
        fnames, labels = read_lines(os.path.join(FLAGS.datadir, 'image-clef/p_list.txt'))
        images = resize_image(fnames)
        labels = np.eye(12)[array(labels).reshape(-1)]

        train_x = images
        test_x = images
        train_y = labels
        test_y = labels

        sz, _ = get_attr_image(FLAGS.nn)

        self.train = DataImage(train_x, train_y, sz, FLAGS.nn, training=True)
        self.test = DataImage(test_x, test_y, sz, FLAGS.nn, training=False)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0


class roomA(object):
    def __init__(self, FLAGS):
        x = np.empty([0, 400, 90])
        y = np.empty([0, 8])
        for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]:
            with open(os.path.join(FLAGS.datadir, 'wifi/processed/roomA/x_2000_60_' + str(i) + '.pkl'), 'rb') as f:
                x_tmp = pickle.load(f)
            with open(os.path.join(FLAGS.datadir, 'wifi/processed/roomA/y_2000_60_' + str(i) + '.pkl'), 'rb') as f:
                y_tmp = pickle.load(f)
            x = np.concatenate((x, x_tmp), axis=0)
            y = np.concatenate((y, y_tmp), axis=0)

        # Delete no activity data
        x = x[y[:, 0] == 0]
        y = y[y[:, 0] == 0]
        y = np.delete(y, 0, axis=1)

        # Data transpose
        x = np.stack(np.split(x, 3, axis=2))
        x = np.transpose(x, (1, 3, 2, 0)).astype(np.float32)

        # Permute the dataset
        permute_idx = np.random.permutation(len(x))
        x = x[permute_idx]
        y = y[permute_idx]

        train_x = x
        test_x = x
        train_y = y
        test_y = y

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0

        print(f"Wifi dataset roomA class is initialized. x.shape: {x.shape}, y.shape: {y.shape}")


class roomB(object):
    def __init__(self, FLAGS):
        x = np.empty([0, 400, 90])
        y = np.empty([0, 8])
        for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]:
            with open(os.path.join(FLAGS.datadir, 'wifi/processed/roomB/x_2000_60_' + str(i) + '.pkl'), 'rb') as f:
                x_tmp = pickle.load(f)
            with open(os.path.join(FLAGS.datadir, 'wifi/processed/roomB/y_2000_60_' + str(i) + '.pkl'), 'rb') as f:
                y_tmp = pickle.load(f)
            x = np.concatenate((x, x_tmp), axis=0)
            y = np.concatenate((y, y_tmp), axis=0)

        # Delete no activity data
        x = x[y[:, 0] == 0]
        y = y[y[:, 0] == 0]
        y = np.delete(y, 0, axis=1)

        # Data transpose
        x = np.stack(np.split(x, 3, axis=2))
        x = np.transpose(x, (1, 3, 2, 0)).astype(np.float32)

        # Permute the dataset
        permute_idx = np.random.permutation(len(x))
        x = x[permute_idx]
        y = y[permute_idx]

        train_x = x[len(x) // 10:]
        test_x = x[:len(x) // 10]
        train_y = y[len(y) // 10:]
        test_y = y[:len(y) // 10]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = None

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = 0

        print(f"Wifi dataset roomB class is initialized. x.shape: {x.shape}, y.shape: {y.shape}")


def read_lines(fname):
    data = open(fname).readlines()
    fnames = []
    labels = []
    for line in data:
        fnames.append(line.split()[0])
        labels.append(int(line.split()[1]))
    return fnames, labels


def resize_image(fnames):
    images = np.ndarray([len(fnames), 256, 256, 3])
    for i in range(len(fnames)):
        img = cv2.imread(fnames[i])
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32)
        img *= 1. / 255
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        images[i] = img
    return images


def normalize_image(image, nn):
    if nn == 'alexnet':
        mean = [0.481, 0.458, 0.409]  # mean of ilsvrc_2012_mean.npy after converting (2, 1, 0)
        image = image - mean
    elif nn == 'resnet':
        mean = [0.485, 0.456, 0.406]  # TODO: should check if this is right
        std = [0.229, 0.224, 0.225]
        image = (image - mean) / std
    else:
        raise ValueError(f"Network {nn} is not compatible with the data.")
    return image


def train_image_process(image, sz, nn='resnet'):
    image = tf.random_crop(image, [tf.shape(image)[0], sz, sz, 3])
    image = tf.image.random_flip_left_right(image)
    image = normalize_image(image, nn)
    return image


def test_image_process(images, sz, nn):
    output = np.ndarray([images.shape[0], sz, sz, images.shape[3]])
    # Central crop
    assert images.shape[1] >= sz and images.shape[2] >= sz
    h_off = (images.shape[1] - sz) // 2
    w_off = (images.shape[2] - sz) // 2
    output = images[:, h_off:h_off + sz, w_off:w_off + sz, :]
    # Normalize
    output = normalize_image(output, nn)
    return output


def get_attr(source, target=None):
    # Processed_attr: [processed size, processed channels, number of classes]
    processed_attr = {
        'usps'     : [28, 1, 10],
        'usps1800' : [28, 1, 10],
        'mnist'    : [28, 1, 10],
        'mnist2000': [28, 1, 10],
        'mnistm'   : [28, 3, 10],
        'svhn'     : [32, 3, 10],
        'syndigits': [32, 3, 10],
        'cifar'    : [32, 3, 9],
        'stl'      : [32, 3, 9],
        'gtsrb'    : [40, 3, 43],
        'synsigns' : [40, 3, 43],
        'amazon'   : [256, 3, 31],
        'webcam'   : [256, 3, 31],
        'dslr'     : [256, 3, 31],
        'c'        : [256, 3, 12],
        'i'        : [256, 3, 12],
        'p'        : [256, 3, 12],
        'roomA'    : [30, 3, 7],
        'roomB'    : [30, 3, 7]
    }

    # Desired_attr: [desired size, desired channels, source normalize, target normalize]
    experiment_attr = {
        'usps_mnist'        : [28, 1],
        'mnist_usps'        : [28, 1],
        'usps1800_mnist2000': [28, 1],
        'mnist2000_usps1800': [28, 1],
        'mnist_mnistm'      : [28, 3],
        'mnistm_mnist'      : [28, 3],
        'svhn_mnist'        : [32, 1],
        'mnist_svhn'        : [32, 1],
        'svhn_syndigits'    : [32, 3],
        'syndigits_svhn'    : [32, 3],
        'cifar_stl'         : [32, 3],
        'stl_cifar'         : [32, 3],
        'synsigns_gtsrb'    : [40, 3],
        'gtsrb_synsigns'    : [40, 3],
        'roomA_roomB'       : [30, 3],
        'roomB_roomA'       : [30, 3]
    }

    if target == None:
        data_sz, data_ch, data_nc = processed_attr[source]
        return data_sz, data_ch, data_nc

    exp = source + '_' + target
    if not exp in experiment_attr:
        raise NotImplementedError("The submitted {} experiment is not valid".format(exp))

    src_sz, src_ch, src_nc = processed_attr[source]
    trg_sz, trg_ch, trg_nc = processed_attr[target]
    exp_sz, exp_ch = experiment_attr[exp]

    return src_sz, trg_sz, exp_sz, src_ch, trg_ch, exp_ch, src_nc


def get_attr_image(nn, source=None):
    exp_ch = 3

    if nn == 'alexnet':
        exp_sz = 227
    elif nn == 'resnet':
        exp_sz = 224
    else:
        raise ValueError(f"Network {nn} is not compatible with the data.")

    if source:
        office_dataset = ['amazon', 'dslr', 'webcam']
        image_clef_dataset = ['i', 'c', 'p']

        if source in office_dataset:
            exp_nc = 31
        elif source in image_clef_dataset:
            exp_nc = 12
        else:
            raise ValueError(f"Dataset {source} is not available.")
        return exp_sz, exp_ch, exp_nc
    else:
        return exp_sz, exp_ch


def get_data(domain, FLAGS):
    return eval(domain)(FLAGS)


def get_info(data, bs):
    print(f"The number of train data is         {data.train_num}")
    print(f"The number of test data is          {data.test_num}")
    print(f"The number of validation data is    {data.val_num}")
    images, labels = data.train.next_batch(bs)
    print(f"The shape of the train batch images is    {images.shape}")
    print(f"The shape of the train batch labels is    {labels.shape}")
    print(f"The max value of the train images is      {np.max(images)}")
    print(f"The min value of the train images is      {np.min(images)}")
    images, labels = data.test.next_batch(bs)
    print(f"The shape of the test batch images is    {images.shape}")
    print(f"The shape of the test batch labels is    {labels.shape}")
    print(f"The max value of the test images is      {np.max(images)}")
    print(f"The min value of the test images is      {np.min(images)}")
