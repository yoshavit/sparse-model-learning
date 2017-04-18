import gym
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from utils import tf_util as U
from utils.save_and_load import get_scope_vars
from utils.cluster_loss import cluster_loss
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
GAMEPLAY_TIMOUT = 300 #moves

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
    def __init__(self, ob_space, ac_space, max_horizon,
                 is_bw = False, latent_dims=10,
                 no_encoder_gradient=False, residual=True):
        """Creates a model-learner framework

        Args:
            max_horizon - number of steps transition model rolls forward (1 means
            1-step lookahead)
        """
        self.replay_memory = []
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.encoder_scope = self.equater_scope = self.transition_scope = None
        self.latent_dims = latent_dims
        self.no_encoder_gradient = no_encoder_gradient
        self.residual = residual
        self.max_horizon = max_horizon
        self.build_inputs()

    def build_inputs(self):
        self.input_states = []
        self.input_actions = []
        self.input_rewards = []
        print("Creating model for %d-step prediction" % self.max_horizon) 
        for t in range(self.max_horizon + 1):
            self.input_states.append(U.get_placeholder("input_state_%d"%t,
                                                        tf.float32, [None] +
                                                        list(self.ob_space.shape)))
            if t != self.max_horizon:
                self.input_rewards.append(U.get_placeholder("input_reward_%d"%t,
                                                        tf.float32, [None]))
                self.input_actions.append(U.get_placeholder("input_action_%d"%t,
                                                             tf.int32, [None]))
                self.input_terminals.append(
                    U.get_placeholder("input_terminal_%d"%t, tf.bool,
                                      [None]))

        self.input_is_valid = U.get_placeholder("input_is_valid", tf.bool,
                                                [None])

    def get_full_inputs(self):
        return self.input_states + self.input_actions + self.input_rewards +\
            self.input_terminals + [self.input_is_valid]

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
        approx_end_state, _ = self.get_future_encoded_state(start_state,
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

    def build_pair_cluster_trainer(self, transition_inds,
                                   pos_weight_multiplier=1,
                                   stepsize=1e-4, name=None):
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
        targets = tf.cast(
            self.input_is_valid,
            tf.float32, name="train_targets")
        with cond_scope(name is not None, tf.variable_scope(name)) as scope:
            update_steps, losses = self.build_clustering_diagnostics(
                encoded_last_states[0], encoded_last_states[1],
                self.input_is_valid,
                stepsize=stepsize,
                pos_weight_multiplier=pos_weight_multiplier)
        return update_steps, losses

    def build_multipair_cumloss_trainer(self,
                                        max_prediction_length,
                                        with_reward_terminal,
                                        pos_weight_multiplier=1,
                                        stepsize=1e-4, name=None):
        targets = tf.cast(
            self.input_is_valid,
            tf.float32, name="train_targets")
        losses = []
        reward_loss_total = tf.constant(0, name="reward_loss")
        terminal_loss_total = tf.constant(0, name="termina_loss")
        state_prediction_loss_totals =\
                [tf.constant(0, name="%d-step_prediction_loss"%t)
                 for t in range(1, max_prediction_length)]
        for t in range(1, max_prediction_length):
            for t0 in range(max_prediction_length - t):
                s0 = self.input_states[t0]
                x0 = self.build_encoder(s0)
                sf = self.input_states[t0+t]
                ac_sequence = self.input_actions[t0:(t0+t)]
                x = x0
                for step in range(t-1): # all but last step
                    x, _, _ = self.build_transition(x, ac_sequence[t0+step])
                # last step
                xf, rew, terminal_logits = self.build_transition(x, ac_sequence)
                xf_true = self.build_encoder(sf)
                state_prediction_loss = self.build_cluster_loss(
                    xf, xf_true, self.input_is_valid, pos_weight_multiplier)
                state_prediction_loss_totals[t-1] += state_prediction_loss
                if with_reward_terminal:
                    true_rew = self.input_rewards[t0+t-1]
                    true_terminal = self.input_terminals[t0+t-1]
                    reward_loss = tf.reduce_mean(tf.boolean_mask(
                        tf.squared_difference(true_rew, rew),
                        self.input_is_valid))
                    reward_loss_total += reward_loss
                    terminal_loss = tf.reduce_mean(self.boolean_mask(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=true_terminal,
                            logits=terminal_logits), self.input_is_valid))
                    terminal_loss_total += terminal_loss
        cumloss = tf.add_n([reward_loss_total, terminal_loss_total] +
                           state_prediction_loss_totals, name="cum_loss")
        losses.append(reward_loss_total); losses.append(terminal_loss_total)
        losses.extend(state_prediction_loss_totals)
        losses.append(cum_loss)
        update_step = tf.train.AdamOptimizer(stepsize).minimize(cum_loss)
        return [update_step], losses


    def build_pair_cluster_executer(self, n_steps):
        start_state = self.input_states[0]
        end_state = self.input_states[n_steps]
        encoded_end_state = self.build_encoder(end_state)
        ac_sequence = self.input_actions[:n_steps]
        approx_end_state, approx_total_reward = self.get_future_encoded_state(
            start_state, ac_sequence)
        input_placeholders = [start_state, end_state] + ac_sequence
        output_placeholders = [encoded_end_state, approx_end_state]
        get_distance = U.function(input_placeholders,
                                  tf.reduce_mean(
                                      tf.squared_difference(
                                          encoded_end_state, approx_end_state),
                                      axis=1))
        get_reward = U.function(input_placeholders, approx_total_reward)
        return [input_placeholders, output_placeholders, get_distance,
                get_reward]

    def build_reward_trainer(self, stepsize=1e-4, name=None):
        state = self.input_states[0]
        action = self.input_actions[0]
        reward = self.input_rewards[0]
        _, estimated_reward = self.get_future_encoded_state(state, [action])
        with cond_scope(name is not None, tf.variable_scope(name)) as scope:
            loss = tf.squared_difference(reward, estimated_reward)
            loss = tf.reduce_mean(tf.boolean_mask(loss, self.input_is_valid),
                                         name="reward_loss")
            update_step = tf.train.AdamOptimizer(stepsize).minimize(loss)
            losses = [loss]; update_steps = [update_step]
        return update_steps, losses

    def build_cluster_loss(self, input1, input2, is_same):
        with tf.variable_scope("cluster_loss") as scope:
            return tf.reduce_mean(cluster_loss(input1, input2, is_same,
                                               pos_weight_multiplier))

    def build_clustering_diagnostics(self, input1, input2, is_same,
                                     train=True, stepsize=DEFAULT_STEPSIZE,
                                     pos_weight_multiplier=1):
        losses = []
        update_steps = []
        if train:
            with tf.variable_scope("cluster_loss") as scope:
                loss = tf.reduce_mean(
                    cluster_loss(input1, input2, is_same,
                                 pos_weight_multiplier))
                losses.append(loss)
                update_steps.append(
                    tf.train.AdamOptimizer(stepsize).minimize(loss))
        with tf.variable_scope("diagnostics") as scope:
            dist = tf.reduce_mean(tf.squared_difference(input1, input2),
                                  axis=1)
            negatives, positives = tf.dynamic_partition(
                dist, tf.cast(is_same, tf.int32), 2)
            if train:
                positive_miss = tf.reduce_mean(pos_weight_multiplier*positives,
                                            name="positive_cluster_loss")
                negative_miss = tf.reduce_mean(tf.exp(-negatives), name="negative_cluster_loss")
                losses.append(positive_miss)
                losses.append(negative_miss)
        return update_steps, losses


    def build_classifier_diagnostics(self,  logits, targets, train=True,
                                     stepsize=DEFAULT_STEPSIZE,
                                     mse_losses_raw=None,
                                     mse_loss_scalar=None,
                                     pos_weight_multiplier=1):
        acceptable_miss = 0.4 # tolerance for prediction error
        losses = []
        update_steps = []
        prob = tf.nn.sigmoid(logits, name="prob")
        if train:
            with tf.variable_scope("class_loss") as scope:
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
                                 x.get_shape().as_list()  ))# + [1]))
            if len(x.get_shape().as_list()) == 3: # black and white
                new_shape = new_shape + [1]
            x = tf.reshape(x, new_shape)
            x = tf.nn.relu(U.conv2d(x, 32, "conv1", filter_size=(5,5),
                                    stride=(2,2)))
            x = tf.nn.relu(U.conv2d(x, 32, "conv2", filter_size=(5,5),
                                    stride=(2,2)))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, "dense1",
                                   weight_init=U.normc_initializer()))
            output = U.dense(x, self.latent_dims, "dense2",
                             weight_init=U.normc_initializer())
        return output

    def build_transition(self, input, action):
        # if residual, adds computed quantity to initial state
        n_factors = 32
        oh_model = True
        x = input
        n_ac = self.ac_space.n
        if not self.transition_scope:
            self.transition_scope = "transition"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.transition_scope, reuse=reuse) as scope:
            self.transition_scope = scope
            if oh_model:
                action = tf.one_hot(action, self.ac_space.n)
                action_factors = U.densenobias(action, n_factors,
                                               "action_factors",
                                               weight_init=U.normc_initializer())
                state_factors = U.densenobias(x, n_factors,
                                              "state_factors",
                                              weight_init=U.normc_initializer())
                factors = tf.multiply(state_factors, action_factors)
                x = U.dense(factors, self.latent_dims, "resulting_state",
                            weight_init=U.normc_initializer())
                reward = tf.squeeze(U.dense(factors, 1, "resulting_reward",
                                 weight_init=U.normc_initializer()), axis=1)
                terminal_logits = tf.squeeze(U.dense(factors, 1,
                                                 "resulting_terminal_logits",
                                                 weight_init=U.normc_initializer()))
            else:
                x = tf.nn.relu(U.dense(x, 1028, "dense1",
                                   weight_init=U.normc_initializer()))
                x = tf.nn.relu(U.dense(x, self.latent_dims*n_ac,
                                       "dense2",
                                       weight_init=U.normc_initializer()))
                x = tf.reshape(x, (-1, n_ac, self.latent_dims))
                mask = tf.cast(tf.one_hot(action, n_ac), tf.bool)
                x = tf.boolean_mask(x, mask)
            if self.residual:
                x += input
            next_state = x
        return next_state, reward, terminal_logits

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

    def get_encoding(self, state):
        if not self.default_encoder:
            self.default_encoder = self.build_encoder(self.input_states[0])
        feed_dict = {self.input_states[0]: np.expand_dims(state, 0)}
        return U.get_session().run(self.default_encoder, feed_dict)

    def get_transition_from_encoding(self, state, action):
        if not self.default_transition:
            self.default_transition = self.build_transition(
                self.input_states[0], self.input_actions[0])
        feed_dict = {self.input_states[0]: np.expand_dims(state, 0),
                     self.input_actions[0]: np.expand_dims(action, 0)}
        next_state, reward, terminal_logits = self.default_transition
        terminal = tf.sigmoid(terminal_logits)
        ns, r, t = U.get_session().run([next_state, reward, terminal],
                                   feed_dict=feed_dict)
        return np.squeeze(ns, 0), np.squeeze(r, 0), np.squeeze(t, 0)

    def get_future_encoded_state(self, state, action_sequence):
        x = self.build_encoder(state)
        for ac in action_sequence:
            x, _, _ = self.build_transition(x, ac)
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
            c = 0
            while not done and c < GAMEPLAY_TIMOUT:
                c += 1
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
                                             as_separate_lists=True,
                                             fraction_from_same_game=0):
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
        transition_seqs = [] # all transition sequences
        per_game_transition_seqs = [] # game-specific transition sequences
        for game in self.replay_memory:
            game_seqs = []
            for i in range(len(game) - n_ac + 1):
                transitions = game[i:i+n_ac]
                states = [transitions[j][0] for j in transition_inds[:-1]] +\
                    [transitions[-1][2]]
                actions = [transitions[j][1] for j in range(len(transitions))]
                rewards = [transitions[j][3] for j in range(len(transitions))]
                terminals = [transitions[j][4] for j in
                             range(len(transitions))]
                transition_seqs.append([states, actions, rewards, terminals])
                game_seqs.append([states, actions, rewards, terminals])
            per_game_transition_seqs.append(game_seqs)
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

            if random.random() < fraction_from_same_game: # Compare frames from the same game
                valid_samples = False
                while not valid_samples:
                    try:
                        true_s_sqs, true_a_sqs, true_r_sqs, true_t_seqs = zip(
                            *random.sample(random.choice(per_game_transition_seqs), n_s))
                        valid_samples = True
                    except ValueError:
                        valid_samples = False
            else: # compare frames from different games
                true_s_sqs, true_a_sqs, true_r_sqs, true_t_sqs = zip(
                    *random.sample(transition_seqs, n_s))
            false_seqs = []
            for i in range(n_s):
                # TODO: create false sequences that are more verifiably wrong
                false_s_seq = [true_s_sqs[(i+j)%n_s][j] for j in range(n_s)]
                true_a_seq = true_a_sqs[i]
                true_r_seq = true_r_sqs[i] # false rewards should be # ignored
                true_t_seq = true_t_sqs[i] # same with terminals
                false_seqs.append([false_s_seq, true_a_seq,
                                   true_r_seq, true_t_seq])

            false_transition_seqs.extend(false_seqs)
        # Need to remove the extra entries to ensure n_false total
        while len(false_transition_seqs) > n_false:
            false_transition_seqs.pop()
        combined_transition_seqs = [(*t, False) for t in
                                false_transition_seqs]

        combined_transition_seqs.extend([(*t, True) for t in
                                     true_transition_seqs])
        if as_separate_lists:
            # Separate each transition sequence into columns
            states = [[combined_transition_seqs[i][0][dim] for i in range(n)]
                      for dim in range(n_s)]
            actions = [[combined_transition_seqs[i][1][dim] for i in range(n)]
                       for dim in range(n_ac)]
            rewards = [[combined_transition_seqs[i][2][dim] for i in range(n)]
                       for dim in range(n_ac)]
            terminals = [[combined_transition_seqs[i][3][dim] for i in
                          range(n)] for dim in range(n_ac)]
            is_valid = [[combined_transition_seqs[i][4] for i in range(n)]]
            combined_transition_seqs = [np.asarray(col) for container in
                                    [states, actions, rewards, terminals, is_valid]
                                    for col in container]
        return combined_transition_seqs


    def visualize_transitions(self, transition_seqs, labels=None,
                              get_distance=None, get_reward=None, action_names=None):
        """
        transition_seqs: states, final_states, ac_seq[0]s, ..., ac_seq[i]s, is_trues
        """
        n_cols = len(transition_seqs)
        n_examples = transition_seqs[0].shape[0]
        if get_distance:
            distances = get_distance(*transition_seqs)
        if get_reward:
            rewards = get_reward(*transition_seqs)
        _, axarr = plt.subplots(n_examples, 2, num=1)
        plt.subplots_adjust(hspace=1)
        for i in range(n_examples):
            before, after, *ac_seq = [transition_seqs[j][i] for j in
                                              range(n_cols)]
            if labels is not None:
                is_true = labels[i]
            plt.sca(axarr[i, 0])
            plt.axis("off")
            ac_string = ""
            for ac in ac_seq:
                if action_names:
                    ac_string += ", " + action_names[ac]
                else:
                    ac_string += ", " + str(ac)
            txt = "Actions: " + ac_string
            if labels is not None:
                txt += ", transition is " + ("TRUE" if is_true else "FALSE")
            if get_distance:
                txt += "\n Estimated distance: " + str(distances[i])
            if get_reward:
                txt += "\n Estimated reward: " + str(rewards[i])
            plt.text(35, 0, txt) # adjust for appropriate spacing
            plt.imshow(before, cmap="Greys")
            plt.title("Before")
            plt.sca(axarr[i, 1])
            plt.axis("off")
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



