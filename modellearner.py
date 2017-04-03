import gym
import gym_mnist
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from utils import tf_util as U
from utils.save_and_load import get_scope_vars
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
DEFAULT_STEPSIZE = 1e-4
MSE_LOSS_SCALAR = 0.1

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

    def __init__(self, ob_space, ac_space, max_horizon, latent_dims=10,
                 no_encoder_gradient=False, residual=True):
        """Creates a model-learner framework

        Args:
            max_horizon - number of steps transition model rolls forward (1 means
            1-step lookahead)
        """
        self.replay_memory = []
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.encoder_scope = self.equater_scope = self.goalcheck_scope = \
            self.transition_scope = None
        self.latent_dims = latent_dims
        self.no_encoder_gradient = no_encoder_gradient
        self.residual = residual
        self.max_horizon = max_horizon
        self.build_inputs()

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

    def build_inputs(self):
        self.input_states = []
        self.input_actions = []
        print("Creating model for %d-step prediction" % self.max_horizon) 
        for t in range(self.max_horizon):
            self.input_states.append(U.get_placeholder("input_state_%d"%t,
                                                        tf.float32, [None] +
                                                        list(self.ob_space.shape)))
            self.input_actions.append(U.get_placeholder("input_action_%d"%t,
                                                         tf.int32, [None]))
        self.input_states.append(U.get_placeholder(
            "input_state_%d"%self.max_horizon, tf.float32, [None] +
            list(self.ob_space.shape)))
        self.input_is_valid = U.get_placeholder("input_is_valid", tf.bool,
                                                [None])
    def get_inputs(self):
        return self.input_states + self.input_actions + [self.input_is_valid]


    def build_pair_classifier_trainer(self, transition_inds,
                                      use_mse_loss=False,
                                      pos_weight_multiplier=1,
                                      stepsize=1e-4, name=None):
        # transition_inds: pair of integers defining the timesteps to be
        # equated
        # name defines the scope of the diagnostics
        assert len(transition_inds) == 2
        Tmax = max(transition_inds)
        Tmin = min(transition_inds)
        assert Tmax <= self.max_horizon and Tmin >= 0
        encoded_last_states = []
        for t in transition_inds:
            s = self.input_states[t]
            ac_sequence = self.input_actions[t:Tmax]
            encoded_last_states.append(
                self.get_future_encoded_state(s, ac_sequence))
        ns_container1, ns_container2 = self.build_shuffler(
            *encoded_last_states)
        train_logits = tf.squeeze(
            self.build_equater(ns_container1, ns_container2), axis=1,
            name="train_logits")
        targets = tf.cast(
            self.input_is_valid,
            tf.float32, name="train_targets")
        with cond_scope(name is not None, tf.variable_scope(name)) as scope:
            mse_losses_raw = tf.squared_difference(ns_container1,
                                               ns_container2)
            update_steps, losses = self.build_classifier_diagnostics(
                train_logits, targets, stepsize=stepsize,
                mse_loss_scalar=MSE_LOSS_SCALAR if use_mse_loss else None,
                mse_losses_raw=mse_losses_raw,
                pos_weight_multiplier=pos_weight_multiplier)
        return update_steps, losses

    def build_pair_classifier_executer(self, n_steps):
        start_state = self.input_states[0]
        end_state = self.input_states[n_steps]
        encoded_end_state = self.build_encoder(end_state)
        ac_sequence = self.input_actions[:n_steps]
        approx_end_state = self.get_future_encoded_state(start_state,
                                                         ac_sequence)
        logits = tf.squeeze(self.build_equater(
            encoded_end_state, approx_end_state),
            axis=1, name="test_logits")
        targets = tf.cast(self.input_is_valid, tf.float32)
        prob, losses = self.build_classifier_diagnostics(
            logits, targets, train=False)
        input_placeholders = [start_state, end_state] + ac_sequence
        get_prob = U.function(input_placeholders, [prob])
        return [input_placeholders, get_prob, losses,
                encoded_end_state, approx_end_state]

    def build_classifier_diagnostics(self, logits, targets, train=True,
                                     stepsize=DEFAULT_STEPSIZE,
                                     mse_losses_raw=None,
                                     mse_loss_scalar=None,
                                     pos_weight_multiplier=1):
        acceptable_miss = 0.4 # tolerance for prediction error
        losses = []
        update_steps = []
        prob = tf.nn.sigmoid(logits, name="prob")
        if train:
            with tf.variable_scope("loss") as scope:
                # Creates classification loss for true/false transitions
                loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    targets,
                    logits,
                    pos_weight_multiplier), name = "classifier_loss")
                losses.append(loss)
                update_steps.append(
                    tf.train.AdamOptimizer(stepsize).minimize(loss))
                if mse_loss_scalar and mse_losses_raw is not None:
                    # Optimizes MSE loss (updates transition function only)
                    true_mse_losses = tf.multiply(
                        tf.reduce_mean(mse_losses_raw, axis=1), targets)
                    mse_loss = tf.reduce_mean(mse_loss_scalar*true_mse_losses,
                                              name="mse_loss")
                    losses.append(mse_loss)
                    transition_vars = get_scope_vars("transition")
                    update_steps.append(
                        tf.train.AdamOptimizer(stepsize).minimize(
                            mse_loss, var_list=transition_vars))
        with tf.variable_scope("diagnostics") as scope:
            acceptable_miss = tf.constant(acceptable_miss)
            negatives, positives = tf.dynamic_partition(
                prob, tf.cast(targets, tf.int32), 2)
            positive_miss = tf.ones_like(positives) - positives
            negative_miss = negatives
            classification_miss = tf.abs(tf.subtract(prob, targets))
            positive_correctly_classified = tf.less(positive_miss,
                                                    acceptable_miss)
            negative_correctly_classified = tf.less(negative_miss,
                                                    acceptable_miss)
            correctly_classified = tf.less(
                classification_miss, acceptable_miss)
            losses.append(tf.reduce_mean(
                tf.cast(correctly_classified, tf.float32),
                name="overall_classification_rate"))
            losses.append(tf.reduce_mean(
                tf.cast(positive_correctly_classified, tf.float32),
                name="positive_classification_rate"))
            losses.append(tf.reduce_mean(
                tf.cast(negative_correctly_classified, tf.float32),
                name="negative_classification_rate"))
            if train:
                losses.append(tf.reduce_mean(-1*pos_weight_multiplier*
                    tf.multiply(positive_miss, tf.log(positive_miss)),
                    name = "positive_classifier_loss"))
                losses.append(tf.reduce_mean(
                    -1*tf.multiply(negative_miss, tf.log(negative_miss)),
                    name = "negative_classifier_loss"))
        if not train:
            return prob, losses
        return update_steps, losses

#------------------------ MODEL SUBCOMPONENTS ----------------------------------

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

    def build_transition(self, input, action):
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
            if self.residual:
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

    def build_equater(self, input1, input2, num_outputs=1):
        if not self.equater_scope:
            self.equater_scope = "equater"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.equater_scope, reuse=reuse) as scope:
            x = tf.concat([input1, input2], 1)
            self.equater_scope = scope
            x = tf.nn.relu(U.dense(x, 128, "dense1",
                                   weight_init=U.normc_initializer()))
            output = U.densenobias(x, num_outputs, "dense2",
                        weight_init=U.normc_initializer())
        return output

    def build_shuffler(self, *input_tensors):
        """Takes a series of tensors, and randomly chooses pairs of them
        (might be the same element twice)
        """
        n_t = len(input_tensors)
        with tf.variable_scope("shuffler") as scope:
            combined_inputs = tf.stack(input_tensors, axis=1)
            indices_shape = tf.stack([tf.shape(combined_inputs)[0],
                                     tf.constant(n_t)])
            random_ints = tf.multinomial(
                tf.log(tf.fill(indices_shape, 10.0)), 1, seed=123)
            # TODO: remove seed
            random_bools1 = tf.squeeze(tf.cast(tf.one_hot(random_ints, n_t),
                                    dtype=tf.bool), axis=1)
            random_bools2 = tf.logical_not(random_bools1)
            out1 = tf.boolean_mask(combined_inputs, random_bools1)
            out2 = tf.boolean_mask(combined_inputs, random_bools2)
        return out1, out2

    # -------------- UTILITIES ------------------------------

    def get_future_encoded_state(self, state, action_sequence):
        x = self.build_encoder(state)
        for ac in action_sequence:
            x = self.build_transition(x, ac)
        return x

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

    def create_true_false_transition_dataset(self, n, fraction_true=0.5,
                                             as_separate_lists=True):
        """ Returns state/action sequences that are true and false for training
        purposes

        Output:
            [[states], [actions], is_valid]
        """

        assert self.replay_memory, "Must gather_gameplay_data before creating\
transition dataset!"
        transition_inds = range(self.max_horizon + 1)
        n_true = int(n*fraction_true)
        n_false = n - n_true
        n_ac = max(transition_inds) # number of actions per sequence
        n_s = len(transition_inds) # number of states per sequence
        transition_seqs = []
        for game in self.replay_memory:
            for i in range(len(game) - n_ac + 1):
                transitions = game[i:i+n_ac]
                states = [transitions[j][0] for j in transition_inds[:-1]] +\
                    [transitions[-1][2]]
                actions = [transitions[j][1] for j in range(len(transitions))]
                transition_seqs.append([states, actions])
                # TODO: add reward/done
        true_transition_seqs = random.sample(transition_seqs, n_true)
        false_transition_seqs = []
        for i in range(n_false//n_s + 1):
            # Create new sequences by mixing previous sequences 
            # Each state is shifted by n times, where n is its index, e.g.
            # (a, b, c)    (a, e, i)
            # (d, e, f) -> (d, h, c)
            # (g, h, i)    (g, b, f)
            # This creates a set of entirely incorrect transition_seqs, with the
            # assumption that a pair of random transitions is not valid
            true_state_seqs, true_action_seqs = zip(
                *random.sample(transition_seqs, n_s))
            false_seqs = []
            for i in range(n_s):
                # TODO: create false sequences that are more verifiably wrong
                false_state_seq = [true_state_seqs[(i+j)%n_s][j] for j in range(n_s)]
                true_action_seq = true_action_seqs[i]
                false_seqs.append([false_state_seq, true_action_seq])
            false_transition_seqs.extend(false_seqs)
        # Need to remove the extra entries to ensure n_false total
        while len(false_transition_seqs) > n_false:
            false_transition_seqs.pop()
        combined_transition_seqs = [(t[0], t[1], False) for t in
                                false_transition_seqs]
        combined_transition_seqs.extend([(t[0], t[1], True) for t in
                                     true_transition_seqs])
        if as_separate_lists:
            # Separate each transition sequence into columns
            states = [[combined_transition_seqs[i][0][dim] for i in range(n)]
                      for dim in range(n_s)]
            actions = [[combined_transition_seqs[i][1][dim] for i in range(n)]
                       for dim in range(n_ac)]
            is_valid = [[combined_transition_seqs[i][2] for i in range(n)]]
            combined_transition_seqs = [np.asarray(col) for container in
                                    [states, actions, is_valid]
                                    for col in container]
        return combined_transition_seqs

    # def create_goalcheck_dataset(self, n)

    def visualize_transitions(self, transition_seqs, labels=None, get_prob=None, action_names=None):
        """
        transition_seqs: states, final_states, ac_seq[0]s, ..., ac_seq[i]s, is_trues
        """
        n_cols = len(transition_seqs)
        n_examples = transition_seqs[0].shape[0]
        if get_prob:
            probs = get_prob(*transition_seqs)[0]
        _, axarr = plt.subplots(n_examples, 2, num=1)
        plt.subplots_adjust(hspace=1)
        for i in range(n_examples):
            before, after, *ac_seq = [transition_seqs[j][i] for j in
                                              range(n_cols)]
            if labels is not None:
                is_true = labels[i]
            plt.sca(axarr[i, 0])
            ac_string = ""
            for ac in ac_seq:
                if action_names:
                    ac_string += ", " + action_names[ac]
                else:
                    ac_string += ", " + str(ac)
            txt = "Actions: " + ac_string
            if labels is not None:
                txt += ", transition is " + ("TRUE" if is_true else "FALSE")
            if get_prob:
                txt += "\n Estimated probability: " + str(probs[i])
            plt.text(30, 0, txt) # adjust for appropriate spacing
            plt.imshow(before, cmap="Greys")
            plt.title("Before")
            plt.sca(axarr[i, 1])
            plt.imshow(after, cmap="Greys")
            plt.title("After")
        plt.pause(0.01)

# taken from http://stackoverflow.com/questions/41695893/tensorflow-conditionally-add-variable-scope
class cond_scope(object):
  def __init__(self, condition, contextmanager):
    self.condition = condition
    self.contextmanager = contextmanager
  def __enter__(self):
    if self.condition:
      return self.contextmanager.__enter__()
  def __exit__(self, *args):
    if self.condition:
      return self.contextmanager.__exit__(*args)

    ''' 
    def build_game_model_cmp_only(self, pos_weight_multiplier=1,
                                  stepsize=DEFAULT_STEPSIZE):
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
        self.logits = self.build_equater(self.encoded_state,
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
        self.build_diagnostics(stepsize)
        '''



