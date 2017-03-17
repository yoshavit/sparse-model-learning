import gym
import gym_mnist
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import os
from termcolor import colored
import scipy.misc
import matplotlib.pyplot as plt
import argparse
from utils import dataset, tf_util as U
#-------# -----------------------------------------------
# # A hacky solution for @yo-shavit to run OpenCV in conda without bugs
# # Others should remove
# import os
# if os.environ['HOME'] == '/Users/yonadav':
    # import sys;
    # sys.path.append("/Users/yonadav/anaconda/lib/python3.5/site-packages")
# #------------------------------------------------------
# import cv2

REPLAY_MEMORY_SIZE = 1500

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

    def __init__(self, ob_space, ac_space, latent_dims=10,
                 no_encoder_gradient=False):
        self.replay_memory = []
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.encoder_scope = self.comparator_scope = self.goalcheck_scope = \
            self.transition_scope = None
        self.latent_dims = latent_dims
        self.no_encoder_gradient = no_encoder_gradient

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

    def build_game_model_inputs(self):
        self.input_state = U.get_placeholder("input_state", tf.float32,
                                             [None] + list(self.ob_space.shape))
        self.input_next_state = U.get_placeholder("input_next_state", tf.float32,
                                                  [None] + list(self.ob_space.shape))
        self.input_action = U.get_placeholder("input_action", tf.int32, [None])
        self.input_is_valid = U.get_placeholder("input_is_valid", tf.bool,
                                                [None])

    def build_game_model_cmp_only(self, pos_weight_multiplier=1):
        """Constructs the necessary infrastructure to train a game model, using
        the simplest architecture: compare two states, and output whether the
        first follows from the second given a certain action.

        Args:
            pos_weight_multiplier: a scalar that determines the bias of the
                classifier towards correct positive predictions by a ratio of
                pos_weight_multiplier:1
        """
        n_ac = self.ac_space.n
        self.build_game_model_inputs()
        self.encoded_state = self.build_encoder(self.input_state)
        self.encoded_next_state = self.build_encoder(self.input_next_state)
        self.logits = self.build_comparator(self.encoded_state,
                                            self.encoded_next_state,
                                            num_outputs=n_ac)
        self.transition_probs = tf.nn.sigmoid(self.logits)
        action_indices = tf.one_hot(self.input_action, n_ac)
        action_logits = tf.reduce_mean(tf.mul(self.logits, action_indices), axis=1)
        # TODO: is boolean_mask more efficient than this trick?
        self.prob = tf.nn.sigmoid(action_logits)
        self.loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                tf.cast(self.input_is_valid, tf.float32),
                action_logits,
                pos_weight_multiplier))
        self.build_diagnostics()

    def build_game_model_with_transition(self, pos_weight_multiplier=1,
                                         residual=True, stepsize=1e-4):
        self.build_game_model_inputs()
        self.encoded_state = self.build_encoder(self.input_state)
        self.encoded_next_state = self.build_encoder(self.input_next_state)
        self.approximated_next_state = self.build_transition(self.encoded_state,
                                                          self.input_action,
                                                          residual=residual)
        # ------------------------------------------------------------------
        # We want to avoid having the comparator learn which of its inputs is
        # the true encoded state and which is the approximation, so we'll
        # randomly swap their positions at runtime
        with tf.variable_scope("discriminator_preprocessing") as scope:
            combined_ns_tensors = tf.stack([self.encoded_next_state,
                                           self.approximated_next_state], axis=1)
            indices_shape = tf.stack([tf.shape(self.encoded_next_state)[0],
                                     tf.constant(2)])
            single_random = tf.squeeze(tf.cast(
                tf.multinomial(tf.log(tf.fill(indices_shape, 10.0)), 1, seed=123),
                tf.bool), axis=1)
            # TODO: remove seed
            single_random_bar = tf.logical_not(single_random)
            normal_or_swapped = tf.stack([single_random, single_random_bar],
                                         axis=1)
            flipped_nos = tf.logical_not(normal_or_swapped)
            ns_container1 = tf.boolean_mask(combined_ns_tensors, normal_or_swapped)
            ns_container2 = tf.boolean_mask(combined_ns_tensors, flipped_nos)
        # ------------------------------------------------------------------
        self.logits = tf.squeeze(
            self.build_comparator(ns_container1, ns_container2), axis=1,
            name="logits")
        self.prob = tf.nn.sigmoid(self.logits, name="prob")
        targets = tf.cast(self.input_is_valid, tf.float32, name="target")
        with tf.variable_scope("loss") as scope:
            self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                targets,
                self.logits,
                pos_weight_multiplier))
        self.build_diagnostics(stepsize, pos_weight_multiplier=pos_weight_multiplier)

    def build_diagnostics(self, stepsize, pos_weight_multiplier=1):
        acceptable_miss = 0.4 # tolerance for prediction error
        self.train_step = tf.train.AdamOptimizer(stepsize).minimize(self.loss)
        with tf.variable_scope("diagnostics") as scope:
            acceptable_miss = tf.constant(acceptable_miss)
            self.get_prob = U.function([self.input_state, self.input_action,
                                        self.input_next_state], [self.prob])
            negatives, positives = tf.dynamic_partition(
                self.prob, tf.cast(self.input_is_valid, tf.int32), 2)
            positive_miss = tf.subtract(tf.ones_like(positives), positives)
            negative_miss = negatives
            classification_miss = tf.abs(tf.subtract(self.prob,
                                         tf.cast(self.input_is_valid, tf.float32)))
            positive_correctly_classified = tf.less(positive_miss, acceptable_miss)
            negative_correctly_classified = tf.less(negative_miss, acceptable_miss)
            self.correctly_classified = tf.less(classification_miss, acceptable_miss)
            self.positive_classification_rate = tf.reduce_mean(
                tf.cast(positive_correctly_classified, tf.float32))
            self.negative_classification_rate = tf.reduce_mean(
                tf.cast(negative_correctly_classified, tf.float32))
            self.classification_rate = tf.reduce_mean(
                tf.cast(self.correctly_classified, tf.float32))
            self.positive_loss = -1*pos_weight_multiplier*tf.reduce_mean(tf.multiply(positive_miss,
                                                            tf.log(positive_miss)))
            self.negative_loss = -1*tf.reduce_mean(tf.multiply(negative_miss,
                                                            tf.log(negative_miss)))

    def build_encoder(self, input):
        x = input
        if not self.encoder_scope:
            self.encoder_scope = "encoder"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.encoder_scope, reuse=reuse) as scope:
            self.encoder_scope = scope
            x = x/255.
            new_shape = list(map(lambda a: a if a!=None else -1,
                                 x.get_shape().as_list() + [1]))
            x = tf.reshape(x, new_shape)
            x = tf.nn.relu(U.conv2d(x, 32, "conv1", filter_size=(5,5)))
            x = tf.nn.relu(U.conv2d(x, 64, "conv2", filter_size=(5,5)))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, "dense1",
                                   weight_init=U.normc_initializer()))
            output = U.densenobias(x, self.latent_dims, "dense2",
                             weight_init=U.normc_initializer())
            if self.no_encoder_gradient:
                output = tf.stop_gradient(output)
        return output

    def build_transition(self, input, action, residual=True):
        # if residual, adds computed quantity to initial state
        x = input
        n_ac = self.ac_space.n
        if not self.transition_scope:
            self.transition_scope = "transition"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.transition_scope, reuse=reuse) as scope:
            self.transition_scope = scope
            x = tf.nn.relu(U.dense(x, 512, "dense1",
                                   weight_init=U.normc_initializer()))
            x = tf.nn.relu(U.dense(x, self.latent_dims*n_ac,
                                   "dense2",
                                   weight_init=U.normc_initializer()))
            x = tf.reshape(x, (-1, n_ac, self.latent_dims))
            mask = tf.cast(tf.one_hot(action, n_ac), tf.bool)
            x = tf.boolean_mask(x, mask)
            if residual:
                x += input
            output = x
        return output

    def build_goalcheck(self, input):
        x = input
        if not self.goalcheck_scope:
            self.goalcheck_scope = "goalcheck"
            reuse = False
        else:
            reuse=True
        with tf.variable_scope(self.goalcheck_scope, reuse=reuse) as scope:
            self.goalcheck_scope = scope
            x = tf.nn.relu(U.dense(x, 128, "dense1",
                                   weight_init=U.normc_initializer()))
            x = tf.nn.sigmoid(U.dense(x, 1, "dense1",
                                      weight_init=U.normc_initializer()))
            output = x
        return output

    def build_comparator(self, input1, input2, num_outputs=1):
        if not self.comparator_scope:
            self.comparator_scope = "comparator"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.comparator_scope, reuse=reuse) as scope:
            x = tf.concat([input1, input2], 1)
            self.comparator_scope = scope
            x = tf.nn.relu(U.dense(x, 128, "dense1",
                                   weight_init=U.normc_initializer()))
            output = U.densenobias(x, num_outputs, "dense2",
                        weight_init=U.normc_initializer())
        return output

    def visualize_mnist_embeddings(self, logdir, target_tensors, sess, data,
                                   labels=None,
                                   vis_mapping=None,
                                   data_placeholders=None,
                                   summary_writer=None):
        """Creates all relevant files to visualize MNIST digit embeddings using
        tensorboard --logdir=LOG_DIR

        Args:
            target_tensors: ?xD tensors containing the desired embedding
            vis_mapping - list of integers, for each in target_tensors the
                index of the relevant data vector
            data - tensors containing data to be fed in 
        """
        print ("Creating embedding")
        for i in range(len(data)): data[i] = np.array(data[i])
        if labels:
            labels = np.array(labels)
        from tensorflow.contrib.tensorboard.plugins import projector
        if not summary_writer:
            summary_writer = tf.summary.FileWriter(logdir)
        if not vis_mapping:
            vis_mapping = [0 for i in range(len(target_tensors))] # use first entry
        config = projector.ProjectorConfig()
        inputs = data
        placeholders = data_placeholders
        embedding_values = do_elementwise_eval(target_tensors, placeholders,
                                               inputs)
        embed_vars = []
        for i in range(len(embedding_values)):
            embed_var = tf.Variable(np.array(embedding_values[i]),
                                    name="layer_%d"%i)
            embed_vars.append(embed_var)
            embed_var.initializer.run()
            embedding = config.embeddings.add()
            embedding.tensor_name = embed_var.name
            embedding.sprite.image_path = os.path.join(logdir,
                                                       'embed_sprite%d.png'%i)
            image_data = data[vis_mapping[i]]
            thumbnail_size = image_data.shape[1]
            embedding.sprite.single_image_dim.extend([thumbnail_size,
                                                      thumbnail_size])
            sprite = images_to_sprite(image_data)
            scipy.misc.imsave(embedding.sprite.image_path, sprite)
        saver = tf.train.Saver(embed_vars)
        saver.save(sess, os.path.join(logdir, 'embed_model.ckpt'))
        if labels is not None:
            embedding.metadata_path = os.path.join(logdir, 'embed_labels.tsv')
            metadata_file = open(embedding.metadata_path, 'w')
            metadata_file.write('Name\tClass\n')
            for ll in range(len(labels)):
                metadata_file.write('%06d\t%d\n' % (ll, labels[ll]))
            metadata_file.close()

        projector.visualize_embeddings(summary_writer, config)
        print("Embedding created.")

    def gather_gameplay_data(self, env, num_games, policy="random"):
        """Collects gameplay transition data and stores it in replay_memory

        Args:
            env: a gym environment used to collect game data
            num_games: total number of games to simulate
            policy: a function mapping from an observation (or None on first
                timestep) to an action. Default is random.
        """
        if not callable(policy):
            policy = lambda obs: env.action_space.sample()

        for i in range(num_games):
            game_memory = []
            obs = env.reset()
            first_step = True
            done = False
            while not done:
                action = policy(obs)
                new_obs, rew, done, _ = env.step(action)
                if not first_step:
                    game_memory.append((obs, action, new_obs, rew, done))
                obs = new_obs
                first_step = False
            self.replay_memory.append(game_memory)
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory = self.replay_memory[-REPLAY_MEMORY_SIZE:]

    def create_true_false_transition_dataset(self, n,
                                                 fraction_true=0.5,
                                                 as_separate_lists=True):
        assert self.replay_memory, "Must gather_gameplay_data before creating\
transition dataset!"
        n_true = int(n*fraction_true)
        n_false = n - n_true
        transitions = [t for game in self.replay_memory for t in game]
        true_transitions = [(t[0], t[1], t[2]) for t in random.sample(
            transitions, n_true)]
        false_transitions = []
        for i in range(n_false//2 + 1):
            t1, t2 = random.sample(transitions, 2)
            tf1 = (t1[0], t1[1], t2[2]) # swap the action results (likely to be incorrect)
            tf2 = (t2[0], t2[1], t1[2])
            false_transitions.extend([tf1, tf2])
        # Need to remove the extra entry or two to ensure num_transitions total
        # transitions
        while len(false_transitions) > n_false:
            false_transitions.pop()
        combined_transitions = [(t[0], t[1], t[2], False) for t in
                                false_transitions]
        combined_transitions.extend([(t[0], t[1], t[2], True) for t in
                                     true_transitions])
        if as_separate_lists:
            combined_transitions = [[combined_transitions[i][dim] for i in
                                     range(n)] for dim in range(4)]
        return combined_transitions

    # def create_goalcheck_dataset(self, n)

    def visualize_transition(self, transitions, sess=None, action_names=None):
        if sess:
            probs = self.get_prob(*transitions[:3])[0]
        _, axarr = plt.subplots(len(transitions), 2, num=1)
        plt.subplots_adjust(hspace=1)
        for i in range(len(transitions)):
            before, action, after, is_true = [transitions[j][i] for j in
                                              range(4)]
            plt.sca(axarr[i, 0])
            if action_names:
                action_name = action_names[action]
            else:
                action_name = str(action)
            txt = "Action: " + action_name + ", transition is " + (
                "TRUE" if is_true else "FALSE")
            if sess:
                txt += "\n Estimated probability: " + str(probs[i])
            plt.text(30, 0, txt) # adjust for appropriate spacing
            plt.imshow(before, cmap="Greys")
            plt.title("Before")
            plt.sca(axarr[i, 1])
            plt.imshow(after, cmap="Greys")
            plt.title("After")
        plt.pause(0.01)

def increment_path(origpath):
    subscript = None
    newpath = origpath
    while True:
        if os.path.exists(newpath):
            if subscript is None:
                newpath = origpath + "--00"
                subscript = 0
            else:
                subscript += 1
                newpath = "%s--%.2i"%(origpath, subscript)
        else:
            break
    return newpath

def load_classic_mnist_data(mnist_env):
    data, labels = mnist_env._get_full_mnist_dataset()
    return data, labels

def do_elementwise_eval(output_tensors, placeholders, inputs):
    """Evaluates the desired tensors using the data/labels, by breaking the
    computation up into batches.
    Args:
        output_tensors - tensors to evaluate and provide in the output
        placeholders - list of input placeholders
        inputs - list of data to be inputted, same order as placeholders
    """
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    if not isinstance(inputs, list):
        inputs = [inputs]
    n = inputs[0].shape[0]
    batch_size = 32
    all_outputs = [[] for tensor in output_tensors]
    for batch_inputs in dataset.iterbatches(inputs, batch_size=batch_size,
                                            shuffle=False):
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[placeholders[i]] = batch_inputs[i]
        for i in range(len(output_tensors)):
            tensor = output_tensors[i]
            output_value = tensor.eval(feed_dict=feed_dict)
            all_outputs[i].append(output_value)
    for k in range(len(all_outputs)):
        all_outputs[k] = np.concatenate(all_outputs[k])
    return all_outputs

# Taken from https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

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
    saver = tf.train.Saver(tf.trainable_variables())
    SAVE_DIR = os.path.join(LOG_DIR, "model.ckpt")

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
            save_path = saver.save(sess, SAVE_DIR)
        if projecting:
            ml.visualize_mnist_embeddings(LOG_DIR, [ml.logits],
                                          sess, [data], labels=labels,
                                          data_placeholders=[ml.digit_input],
                                          labels_placeholder=ml.digit_labels,
                                          )
        sw.close()
        print("To visualize results, call:")
        print('tensorboard --logdir=%s' % LOG_DIR)

def __main__():
    parser = argparse.ArgumentParser(description="Trains a model_learner on the MNIST game.")
    parser.add_argument('--name', default="default", help="Name of current experiment.")
    parser.add_argument('--load',
                        help="Path to a folder containing the checkpoint for a network with the same\
architecture, relative to file")
    parser.add_argument('--load-encoder',
                        help="Path to a data file for the encoder only")
    parser.add_argument('--project', action="store_true",
                        help="whether to generate a latent-space projection")
    parser.add_argument('--train', action="store_true", help="Train the model")
    parser.add_argument('-batchsize', help="Batch size for learning", type=int,
                        default=64)
    parser.add_argument('--stationary', action="store_true",
                        help="Whether to use a fixed or varying game experience distribution for training")
    parser.add_argument('--visualize', action="store_true",
                        help="Display a few sample predictions after each epoch")
    parser.add_argument('-epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('-stepsize', type=float, default=1e-3, help="train step size")

    args = parser.parse_args()
    CUSTOM_NAME = args.name
    #TODO WORK IN LOAD_ENCODER
    if args.load:
        reuse_weights = True
        OLD_SAVE_PATH = os.path.join(args.load, 'model.ckpt')
    else:
        reuse_weights = False
    training = args.train
    projecting = args.project
    stationary_distribution = args.stationary
    visualize = args.visualize
    num_epochs = args.epochs
    batch_size = args.batchsize
    stepsize = args.stepsize
    summarize = True
    no_encoder_gradient = False
    num_games = 500
    n = 10000

    BASE_DIR = os.path.join("./data", CUSTOM_NAME)
    LOG_DIR = increment_path(os.path.join(BASE_DIR, "logdir"))
    SAVE_DIR = os.path.join(LOG_DIR, "model.ckpt")
    print("Writing results to " + LOG_DIR)
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                      mnist_env.action_space,
                      latent_dims=10,
                      no_encoder_gradient=no_encoder_gradient)
    ml.build_game_model_with_transition(pos_weight_multiplier=3,
                                        stepsize=stepsize)
    ml.gather_gameplay_data(mnist_env, num_games)
    transitions = ml.create_true_false_transition_dataset(
                n, fraction_true=1/3)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    placeholders = [ml.input_state, ml.input_action,
                    ml.input_next_state,
                    ml.input_is_valid]
    losses = [ml.loss,  ml.positive_loss,
              ml.negative_loss, ml.positive_classification_rate,
              ml.classification_rate, ml.negative_classification_rate]
    loss_names = ["overall loss",
                  colored("positive loss", "red"),
                  colored("negative loss","cyan"),
                  "overall accuracy",
                  colored("positive accuracy", "red"),
                  colored("negative accuracy", "cyan")]
    performupdate = U.function(placeholders, losses + [ml.train_step])

    with tf.Session() as sess:
        sw = tf.summary.FileWriter(LOG_DIR, sess.graph)
        init.run()
        if reuse_weights:
            saver.restore(sess, OLD_SAVE_PATH)
            print("Loaded previous weights")
        if not tf.gfile.Exists(SAVE_DIR):
            tf.gfile.MakeDirs(SAVE_DIR)
        if training:
            for i in range(num_epochs):
                print ("Epoch %d: "%i)
                if not stationary_distribution:
                    ml.gather_gameplay_data(mnist_env, num_games)
                    transitions = ml.create_true_false_transition_dataset(
                                n, fraction_true=1/3)
                batch_losses = np.zeros(len(losses))
                l = 0
                for batch in tqdm(dataset.iterbatches(transitions,
                                          batch_size=batch_size, shuffle=True)):
                    *losses, _ = performupdate(*batch)
                    batch_losses += np.array(losses)
                    l +=1
                batch_losses = batch_losses/l
                for name, loss in zip(loss_names, batch_losses):
                    print(name + ": %.5f"%loss)
                if visualize:
                    t_inds = random.sample(range(n), 4)
                    t = [[transitions[i][j] for j in t_inds] for i in range(4)]
                    plt.clf()
                    ml.visualize_transition(
                        t, sess=sess, action_names=mnist_env.get_action_meanings())
                saver.save(sess, SAVE_DIR)
        if projecting:
            ml.visualize_mnist_embeddings(
                LOG_DIR,
                [ml.encoded_state, ml.encoded_next_state, ml.approximated_next_state],
                sess,
                [transitions[0], transitions[1], transitions[2]],
                summary_writer=sw,
                labels=transitions[3],
                vis_mapping=[0, 2, 2],
                data_placeholders=[ml.input_state, ml.input_action, ml.input_next_state]
            )
        sw.close()
        print("To visualize results, call:")
        print('tensorboard --logdir=%s' % LOG_DIR)

def debug_visualize_transition(reuse_weights=True):
    LOG_DIR = "./data/logdir"
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                                 mnist_env.action_space)
    ml.build_game_transition_classifier_cmp_only(pos_weight_multiplier=3)
    ml.gather_gameplay_data(mnist_env, 100)
    n = 1000
    transitions = ml.create_true_false_transition_dataset(
                n, fraction_true=1/3)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    save_dir = os.path.join(LOG_DIR, 'stationary_model')
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')

    with tf.Session() as sess:
        init.run()
        if reuse_weights:
            saver.restore(sess, checkpoint_path)
        while True:
            t_inds = random.sample(range(n), 5)
            t = [[transitions[i][j] for j in t_inds] for i in range(4)]
            ml.visualize_transition(t, sess=sess,
                                    action_names=mnist_env.get_action_meanings())
            plt.pause(1)
            plt.clf()

# debug_visualize_transition(False)
__main__()
# train_classic_mnist()




