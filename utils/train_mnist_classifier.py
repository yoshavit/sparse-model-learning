raise RuntimeError("Not implemented, need to glue these parts together.")
import gym
import gym_mnist
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import argparse
from utils import dataset, tf_util as U
from modellearner import *
from utils.visualize_embeddings import visualize_embeddings
from utils.save_and_load import save_scope, load_scope, get_scope_vars

# ------------------------------------------------------------------
def build_mnist_classifier_model(self, stepsize):
     self.digit_input = U.get_placeholder("digit_input", tf.float32,
                                         [None] + list(
                                             self.ob_space.shape))
     self.logits = self.build_encoder(self.digit_input)
     self.digit_labels = U.get_placeholder("digit_label",
                                          tf.int32,
                                          [None])
     self.loss = tf.reduce_mean(
         tf.nn.sparse_softmax_cross_entropy_with_logits(
         logits=self.logits, labels=self.digit_labels))
     self.train_step = tf.train.AdamOptimizer(stepsize).minimize(self.loss)
     self.correctly_classified = tf.cast(tf.equal(tf.argmax(self.logits, axis=1),
         tf.cast(self.digit_labels, tf.int64)), dtype=tf.float32)
     self.classification_rate = tf.reduce_mean(self.correctly_classified)
# ------------------------------------------------------------------
def train_classic_mnist():
    CUSTOM_NAME = "ideal_encoder"
    BASE_DIR = os.path.join("./data", CUSTOM_NAME)
    LOG_DIR = increment_path(os.path.join(BASE_DIR, "logdir"))
    training = True
    reuse_weights = False
    projecting = True
    num_epochs = 1
    batch_size = 64
    if reuse_weights:
        custom_weight_dir = os.path.join(BASE_DIR, "logdir")
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                                 mnist_env.action_space, latent_dims=10)
    ml.build_mnist_classifier_model(stepsize=1e-3)
    digit_data, digit_labels = load_classic_mnist_data(mnist_env)
    pairs = list(zip(digit_data, digit_labels))[:10000]
    data, labels = zip(*pairs)
    init = tf.global_variables_initializer()

    placeholders = [ml.digit_input, ml.digit_labels]
    losses = [ml.loss, ml.classification_rate]
    performupdate = U.function(placeholders, losses +
                               [ml.train_step])

    print("Training an encoder on the original MNIST dataset.")
    with tf.Session() as sess:
        init.run()
        sw = tf.summary.FileWriter(LOG_DIR, sess.graph)
        if reuse_weights:
            saver.restore(sess, custom_weight_dir)
            print("Loaded previous weights")
        if training:
            for i in range(num_epochs):
                print ("Epoch %d: "%i)
                batch_losses = np.zeros(len(losses))
                l=0
                for batch in tqdm(dataset.iterbatches([data, labels],
                                                      batch_size=batch_size,
                                                      shuffle=True)):
                    *losses, _ = performupdate(*batch)
                    batch_losses += np.array(losses)
                    l += 1
                batch_loss, batch_classification_rate = batch_losses/l
                print ("Log loss: ~%0.5f, training accuracy: ~%0.5f" %(
                    batch_loss, batch_classification_rate))
        if projecting:
            visualize_embeddings(LOG_DIR, [ml.logits],
                                          sess, [data], labels=labels,
                                          data_placeholders=[ml.digit_input],
                                          labels_placeholder=ml.digit_labels,
                                          )
        sw.close()
        print("To visualize results, call:")
        print('tensorboard --logdir=%s' % LOG_DIR)


