import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import tf_util as U
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

REPLAY_MEMORY_SIZE = 1000
DEFAULT_STEPSIZE = 1e-4
MSE_LOSS_SCALAR = 0.1
GAMEPLAY_TIMOUT = 100 #moves

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
                 latent_dims=10,
                 residual=True):
        """Creates a model-learner framework

        Args:
            max_horizon - number of steps transition model rolls forward (1 means
            1-step lookahead)
        """
        self.replay_memory = []
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.encoder_scope = self.transition_scope = None
        self.default_encoder = self.default_transition = None
        self.latent_dims = latent_dims
        self.residual = residual
        self.max_horizon = max_horizon
        self.build_inputs()

    def build_inputs(self):
        self.input_states = []
        self.input_actions = []
        self.input_rewards = []
        self.input_terminals = []
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
        self.input_latent_state = U.get_placeholder("input_latent_state_0",
                                                    tf.float32,
                                                    [None, self.latent_dims])

    def get_full_inputs(self):
        return self.input_states + self.input_actions + self.input_rewards +\
            self.input_terminals

    def build_multipair_cumloss_trainer(self,
                                        max_prediction_length,
                                        with_reward_terminal,
                                        pos_weight_multiplier=1,
                                        stepsize=1e-4, name=None,
                                        reward_scalar=1,
                                        terminal_scalar=1):
        losses = []
        reward_loss_total = 0
        terminal_loss_total = 0
        state_prediction_loss_totals = [0 for t in range(max_prediction_length)]
        for t in range(1, max_prediction_length + 1):
            for t0 in range(max_prediction_length - t + 1):
                s0 = self.input_states[t0]
                x0 = self.build_encoder(s0)
                rfm1 = self.input_rewards[t0+t-1]
                tfm1 = self.input_terminals[t0+t-1]
                sf = self.input_states[t0+t]
                xf = self.build_encoder(sf)
                ac_sequence = self.input_actions[t0:(t0+t)]
                x = x0
                for step in range(t-1): # all but last step
                    x, _, _ = self.build_transition(x, ac_sequence[step])
                xf_hat, rfm1_hat, tfm1_hat_logits = self.build_transition(
                    x, ac_sequence[t-1]) # last step
                state_prediction_loss = tf.reduce_mean(tf.squared_difference(
                    xf, xf_hat))
                state_prediction_loss_totals[t-1] += state_prediction_loss
                if with_reward_terminal:
                    reward_loss = tf.reduce_mean(
                        tf.squared_difference(rfm1, rfm1_hat))
                    reward_loss_total += reward_loss
                    terminal_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.cast(tfm1, tf.float32),
                            logits=tfm1_hat_logits))
                    terminal_loss_total += terminal_loss
        reward_loss_total = tf.identity(reward_scalar*reward_loss_total,
                                        name="reward_loss")
        terminal_loss_total = tf.identity(terminal_scalar*terminal_loss_total,
                                          name="terminal_loss")
        state_prediction_loss_totals = [tf.identity(
            state_prediction_loss_totals[i], name="%d-step_prediction_loss"%(i+1))
            for i in range(max_prediction_length)]
        cumloss = tf.add_n([reward_loss_total, terminal_loss_total] +
                           state_prediction_loss_totals, name="cum_loss")
        losses.append(reward_loss_total); losses.append(terminal_loss_total)
        losses.extend(state_prediction_loss_totals)
        losses.append(cumloss)
        update_step = tf.train.AdamOptimizer(stepsize).minimize(cumloss)
        return [update_step], losses


    def build_pair_cluster_executer(self, n_steps):
        start_state = self.input_states[0]
        end_state = self.input_states[n_steps]
        encoded_end_state = self.build_encoder(end_state)
        ac_sequence = self.input_actions[:n_steps]
        approx_end_state = self.build_future_encoded_state(
            start_state, ac_sequence)
        input_placeholders = [start_state, end_state] + ac_sequence
        output_placeholders = [encoded_end_state, approx_end_state]
        get_distance = U.function(input_placeholders,
                                  tf.reduce_mean(
                                      tf.squared_difference(
                                          encoded_end_state, approx_end_state),
                                      axis=1))
        return [input_placeholders, output_placeholders, get_distance]

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
        n_factors = 512
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
                x = tf.tanh(U.dense(factors, self.latent_dims, "resulting_state",
                            weight_init=U.normc_initializer()))
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

    def build_future_encoded_state(self, state, action_sequence):
        x = self.build_encoder(state)
        for ac in action_sequence:
            x, _, _ = self.build_transition(x, ac)
        return x

    # -------------- UTILITIES ------------------------------

    def get_encoding(self, state):
        if self.default_encoder is None:
            self.default_encoder = self.build_encoder(self.input_states[0])
        feed_dict = {self.input_states[0]: np.expand_dims(state, 0)}
        return U.get_session().run(self.default_encoder, feed_dict)

    def get_transition_from_encoding(self, latent_state, action):
        if self.default_transition is None:
            self.default_transition = self.build_transition(
                self.input_latent_state, self.input_actions[0])
        if latent_state.ndim == 1:
            latent_state = np.expand_dims(latent_state, 0)
        action = np.array(action)
        if action.ndim == 0:
            action = np.expand_dims(action, 0)
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_actions[0]: action}
        next_latent_state, reward, terminal_logits = self.default_transition
        terminal = tf.sigmoid(terminal_logits)
        nx, r, t = U.get_session().run([next_latent_state, reward, terminal],
                                   feed_dict=feed_dict)
        print (nx, r, t)
        return np.squeeze(nx, 0), np.squeeze(r, 0), np.squeeze(t, 0)

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
            done = False
            c = 0
            while not done and c < GAMEPLAY_TIMOUT:
                c += 1
                action = policy(obs)
                new_obs, rew, done, info = env.step(action)
                game_memory.append((obs, action, new_obs, rew, bool(done), info))
                obs = new_obs
            self.replay_memory.append(game_memory)
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory = self.replay_memory[-REPLAY_MEMORY_SIZE:]

    def visualize_transitions(self, transition_seqs, labels=None,
                              get_distance=None,
                              get_reward=None,
                              get_feature=None,
                              get_next_feature=None,
                              visualize_feature=None,
                              action_names=None):
        """
        transition_seqs: states, final_states, ac_seq[0]s, ..., ac_seq[i]s, is_trues
        """
        n_cols = len(transition_seqs)
        n_examples = transition_seqs[0].shape[0]
        if get_distance:
            distances = [get_feature(seq) for seq in transition_seqs]
        if get_reward:
            rewards = [get_feature(seq) for seq in transition_seqs]
        if get_feature:
            features_before = [get_feature(seq[0]) for seq in transition_seqs]
            features_after = [get_feature(seq[1]) for seq in transition_seqs]
        if get_next_feature:
            features_after = [get_next_feature(seq) for seq in transition_seqs]
        _, axarr = plt.subplots(n_examples, 2, num=1)
        plt.subplots_adjust(hspace=1)
        for i in range(n_examples):
            before, after, *ac_seq = [transition_seqs[j][i] for j in
                                      range(n_cols)]
            # if visualize_feature:
                # before = visualize_feature(before, features_before[i])
                # after = visualize_feature(after, features_after[i])
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
            visualize_feature(features_before[i])
            plt.title("Before")
            plt.sca(axarr[i, 1])
            plt.axis("off")
            plt.imshow(after, cmap="Greys")
            visualize_feature(features_after[i])
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

