import gym
import gym_mnist
import numpy as np
import tensorflow as tf
import tf_util as U
from tqdm import tqdm
import random
import os
#-------# -----------------------------------------------
# # A hacky solution for @yo-shavit to run OpenCV in conda without bugs
# # Others should remove
# import os
# if os.environ['HOME'] == '/Users/yonadav':
    # import sys;
    # sys.path.append("/Users/yonadav/anaconda/lib/python3.5/site-packages")
# #------------------------------------------------------
# import cv2


"""
Create a class that takes env as input
Repeatedly acts in the environment, aggregates a set of transitions
Constructs the complete pipeline
Trains by drawing true samples, and creating random (statistically incorrect)
ones.
Then can take input, action, and yield lower-dimensional output state

Quantities I need to know:
    Image dimensions
    Action-space dimension
    Number of hidden layers for each variable
"""

class ModelLearner:

    def __init__(self, ob_space, ac_space, latent_dims=10):
        self.replay_memory = []
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.encoder_intialized = False
        self.latent_dims = latent_dims

    # def build_model(self):
        # self.state_input = U.get_placeholder("input_state", tf.float32,
                                             # self.ob_space.shape)
        # self.action_input = U.get_placeholder("input_action", tf.float32,
                                              # self.ac_space.n)
        # self.next_state_input = U.get_placeholder("input_next_state",
                                                  # tf.float32,
                                                  # self.ob_space.shape)
        # self.build_encoder()
        # self.build_transition()
        # self.build_discriminator()
        # self.build_trainer()

    def build_mnist_classifier_model(self):
         self.digit_input = U.get_placeholder("digit_input", tf.float32,
                                             [None] + list(
                                                 self.ob_space.shape))
         logits = self.build_encoder(self.digit_input)
         self.digit_labels = U.get_placeholder("digit_label",
                                              tf.int32,
                                              [None])
         self.loss = tf.reduce_mean(
             tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits, self.digit_labels))
         self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
         self.classification_rate = tf.reduce_mean(tf.cast(tf.equal(
             tf.argmax(logits, axis=1),
             tf.cast(self.digit_labels, tf.int64)), dtype=tf.float32))



    def build_encoder(self, input):
        x = input
        with tf.variable_scope("encoder") as scope:
            if self.encoder_intialized: scope.reuse_variables()
            self.encoder_intialized = True
            x = x/255.
            new_shape = list(map(lambda a: a if a!=None else -1, 
                                 x.get_shape().as_list() + [1]))
            x = tf.reshape(x, new_shape)
            x = tf.nn.relu(U.conv2d(x, 32, "conv1", filter_size=(5,5)))
            x = tf.nn.relu(U.conv2d(x, 64, "conv2", filter_size=(5,5)))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 1024, "dense1",
                                   weight_init=U.normc_initializer()))
            output = U.dense(x, self.latent_dims, "dense2",
                             weight_init=U.normc_initializer())
        return output

    def gather_gameplay_data(self, env, num_games, policy="random"):
        if not callable(policy):
            policy = lambda obs: env.action_space.sample()

        for i in range(num_games):
            game_memory = []
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs)
                new_obs, rew, done, _ = env.step(action)
                game_memory.append((obs, action, new_obs, rew, done))
                obs = new_obs
            self.replay_memory.append(game_memory)

def load_classic_mnist_data(mnist_env):
    data, labels = mnist_env._get_full_mnist_dataset()
    return data, labels

def __main__():
    num_epochs = 121
    steps_per_epoch = 10
    batch_size = 32
    mnist_env = gym.make("mnist-v0")
    model_learner = ModelLearner(mnist_env.observation_space,
                                 mnist_env.action_space)
    model_learner.build_mnist_classifier_model()
    digit_data, digit_labels = load_classic_mnist_data(mnist_env)
    pairs = list(zip(digit_data, digit_labels))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    save_path = "/tmp/model.cpkt"
    with tf.Session() as sess:
        init.run()
        if os.path.isfile(save_path):
            saver.restore(sess, save_path)
            print("Loaded previous weights...")
        for i in range(num_epochs):
            print ("Epoch %d: "%i)
            for j in tqdm(range(steps_per_epoch)):
                batch_data, batch_labels = zip(
                    *random.sample(pairs, batch_size))
                feed_dict = {model_learner.digit_input: batch_data,
                             model_learner.digit_labels: batch_labels}
                model_learner.train_step.run(feed_dict=feed_dict)
            batch_loss = model_learner.loss.eval(feed_dict=feed_dict)
            batch_classification_rate = model_learner.classification_rate.eval(feed_dict=feed_dict)
            print ("Log loss: ~%0.4f, training accuracy: ~%0.4f" %(
                batch_loss, batch_classification_rate))
            if i%30 == 0:
                save_path = saver.save(sess, save_path)
                test_accuracy = model_learner.classification_rate.eval(
                    feed_dict={model_learner.digit_input: digit_data,
                    model_learner.digit_labels: digit_labels})
                print("Overall test accuracy: %0.5f" % test_accuracy)

__main__()




