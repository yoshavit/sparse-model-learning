import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2

N_FINAL_FILTERS = 64

def create_negative_data(input_data):
    # input_data is n x (side_length x side_length) array of mnist digits
    if len(input_data.shape) == 2:
        sidelength = int(np.sqrt(input_data.shape[1]))
        input_data = np.reshape(input_data, [-1, sidelength, sidelength])
    quadrants = [np.hsplit(a, 2) for a in np.dsplit(input_data, 2)]
    for i in range(2):
        for j in range(2):
            q = np.random.permutation(quadrants[i][j])
            quadrants[i][j] = np.transpose(np.rot90(np.transpose(q, [1,2,0]), np.random.randint(4)), [2,0,1])
    output = np.dstack([np.hstack(quadrants[i]) for i in [0,1]])
    return output

def synthesize_batch(raw_data, raw_labels, neg_to_pos_ratio=0.5):
    n_pos = raw_labels.shape[0]
    n_neg = int(n_pos*neg_to_pos_ratio)
    assert n_neg < n_pos, "Can't have more than half the data be negative"
    positive_labels = np.concatenate([raw_labels, np.zeros([n_pos, 1])], axis=1)
    positive_data = np.reshape(raw_data, [-1, 28, 28])
    negative_labels = np.zeros([n_neg, 11]); negative_labels[:,10] = 1
    negative_data = create_negative_data(positive_data)[:n_neg,:,:]
    labels = np.concatenate([positive_labels, negative_labels], axis=0)
    data = np.concatenate([positive_data, negative_data], axis=0)
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return data, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("W", initializer=initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable("b", initializer=initial)

def conv2d(x, name, shape, strides=[1,1,1,1]):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        output = tf.nn.conv2d(x, W, padding="SAME", strides=[1,1,1,1])
        print("Conv2d weight name: {}".format(W.name))
    return output

def fc_layer(x, name, output_size, input_shape=None):
    with tf.variable_scope(name):
        if input_shape:
            in_x, in_y, in_filt = input_shape
        else:
            in_x, in_y, in_filt = map(int, x.get_shape()[1:])
        W = weight_variable([in_x, in_y, in_filt, output_size])
        b = bias_variable([1, 1, 1, output_size])
        output = tf.nn.conv2d(x, W, padding="VALID", strides=[1,1,1,1]) + b
        output += b
        print("FC weight name: {}".format(W.name))
    return output, (in_x, in_y, in_filt)

def mnist_fcn(a, reuse=False, fc_input_shape=None):
    with tf.variable_scope("fcn", reuse=reuse):
        z = tf.expand_dims(a, axis=-1)
        z = tf.nn.relu(conv2d(z, "conv1", [4, 4, 1, 32], strides=[1,2,2,1]))
        z = tf.nn.relu(conv2d(z, "conv2", [4, 4, 32, N_FINAL_FILTERS], strides=[1,2,2,1]))
        #z, fc_input_shape = fc_layer(z, "fc1", 10, input_shape=fc_input_shape)
        z, fc_input_shape = fc_layer(z, "fc1", 11, input_shape=fc_input_shape)
        output = tf.nn.softmax(z)
        logits = z
    return output, logits, fc_input_shape

def train_mnist_fcn(logdir, restart=False, n_steps=2000):
    if not tf.gfile.Exists(logdir):
        print("Making {}".format(logdir))
        tf.gfile.MakeDirs(logdir)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, name="data", shape=[None, 28, 28])
    # y_ = tf.placeholder(tf.float32, name="label", shape=[None, 10])
    y_ = tf.placeholder(tf.float32, name="label", shape=[None, 11])
    _, y_logits, params = mnist_fcn(x)
    y_logits = tf.squeeze(y_logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    pct_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_, 1)), tf.float32))
    saver = tf.train.Saver()
    savepath = os.path.join(logdir, "model.ckpt")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latest_checkpoint = tf.train.latest_checkpoint(logdir)
        print(latest_checkpoint)
        if latest_checkpoint is not None and not restart:
            print("restoring fcn weights")
            saver.restore(sess, latest_checkpoint)
        for i in range(n_steps):
            X, Y = mnist.train.next_batch(16)
            X, Y = synthesize_batch(X, Y)
            feed_dict = {x: X, y_: Y}
            sess.run(train_op, feed_dict=feed_dict)
            if i%50 == 0:
                X, Y = synthesize_batch(*mnist.train.next_batch(256))
                feed_dict = {x: X, y_: Y}
                batch_loss, batch_pct_correct = sess.run([loss, pct_correct], feed_dict=feed_dict)
                print("Train cross-Entropy Loss: {:.3f}, Train accuracy: {:.3f}".format(batch_loss, batch_pct_correct))
                saver.save(sess, savepath)
                print("Saved weights at {}".format(savepath))
        X, Y = synthesize_batch(*mnist.train.next_batch(512))
        feed_dict = {x: X, y_: Y}
        batch_loss, batch_pct_correct = sess.run([loss, pct_correct], feed_dict=feed_dict)
        print("Test cross-Entropy Loss: {:.3f}, Test accuracy: {:.3f}".format(batch_loss, batch_pct_correct))
    return mnist, params

def create_hybrid_images(mnist_images, dx=3, dy=3):
    mnist_images = np.reshape(mnist_images, [-1, 28, 28])
    n = len(mnist_images)
    #mnist_images = mnist_images[:,::2,::2]
    hybrid_images = []
    for i in range(n//(dx*dy)):
        hi_y = []
        for a in range(dx):
            hi_y.append(np.concatenate([mnist_images[i + a*dy + b] for b in range(dy)], axis=1))
        hi = np.concatenate(hi_y, axis=0)
        hybrid_images.append(hi)
    return np.stack(hybrid_images, 0)

def test_mnist_fcn(logdir, fcn_params, mnist=None):
    if mnist is None:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    compound_placeholder = tf.placeholder(tf.float32, name="input_multiimage", shape=[None, 84, 84])
    compound_res, compound_res_logits, _ = mnist_fcn(compound_placeholder, reuse=True, fc_input_shape=fcn_params)
    compound_res = tf.nn.max_pool(compound_res, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, logdir)
        data, _ = mnist.train.next_batch(9)
        compound_data = create_hybrid_images(data)
        result = sess.run(compound_res, feed_dict={compound_placeholder:compound_data})

    fig, axes = plt.subplots(4,3, figsize=(6,6))
    axes[0][0].imshow(compound_data[0,:,:], cmap="Greys")
    for i in range(10):
        indx = (i+1)//3
        indy = (i+1)%3
        padded_result = result[0,:,:,i]
        aximg = axes[indx][indy].imshow(padded_result, cmap="Blues")
        fig.colorbar(aximg, ax=axes[indx][indy])
        aximg.set_clim(0,1)
        axes[indx][indy].get_xaxis().set_visible(False)
        axes[indx][indy].get_yaxis().set_visible(False)
        axes[indx][indy].set_title("Class {}".format(i))
    plt.show()
    cv2.waitKey(0)

class MNISTFeatureExtractor:
    def __init__(self, logdir, sess=None, imshape=[84,84],
                 fcn_params=[28,28,N_FINAL_FILTERS],
                 train=False):
        #imshape is the 2d dimensions of the input images
        self.logdir = logdir
        if train:
            _, fcn_params = train_mnist_fcn(logdir)
            print(fcn_params)
        else:
            assert fcn_params, "if not training must specify FCN shape"
        self.fcn_params = fcn_params
        self.sess = sess if sess else tf.Session()
        self.input_placeholder = tf.placeholder(tf.float32,
                                                name="input_images",
                                                shape=[None] + imshape)
        output, _, _ = mnist_fcn(self.input_placeholder, reuse=train,
                                      fc_input_shape=fcn_params)
        self.output = tf.nn.max_pool(output, ksize=[1,8,8,1],
                                     strides=[1,8,8,1], padding="SAME")
        self.saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)
        self.saver.restore(self.sess, logdir)

    def run_fcn(self, images):
        if images.ndim == 2 or images.shape[2] == 1:
            # either no channel and batch dimensions, or
            # a "channels" dimension but no batch
            images = np.expand_dims(images, axis=0)
        if len(images.shape) == 4:
            assert images.shape[3] == 1
            images = np.squeeze(images, axis=3)
        with self.sess as sess:
            result = sess.run(self.output,
                              feed_dict={self.input_placeholder: images})
        return result


def __main__():
    logdir = "weights/mnist_fcn"
    # mnist, params = train_mnist_fcn(logdir, restart=False, n_steps=300)
    fe = MNISTFeatureExtractor(logdir, train=True)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data, labels = mnist.train.next_batch(18)
    examples = create_hybrid_images(data)
    f = fe.run_fcn
    result = f(examples)
    fig, axes = plt.subplots(4,3, figsize=(6,6))
    print("plotting result")
    axes[0][0].imshow(examples[0,:,:], cmap="Greys")
    for i in range(10):
        indx = (i+1)//3
        indy = (i+1)%3
        padded_result = result[0,:,:,i]
        #padded_result = np.pad(result_logits[0,:,:,i], [[7,6],[7,6]], 'constant', constant_values=0)
        aximg = axes[indx][indy].imshow(padded_result)
        fig.colorbar(aximg, ax=axes[indx][indy])
        aximg.set_clim(0,1)
        axes[indx][indy].get_xaxis().set_visible(False)
        axes[indx][indy].get_yaxis().set_visible(False)
        axes[indx][indy].set_title("Class {}".format(i))
    plt.show()
    cv2.waitKey(0)

__main__()





