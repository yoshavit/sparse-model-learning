import numpy as np
import tensorflow as tf
from utils import tf_util as U
from utils.gruacell import GRUACell
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

class EnvModel:
    def __init__(self, ob_space, ac_space, feature_size, latent_size=128):
        """Creates a model-learner framework
        """
        self.replay_memory = []
        self.ob_space = list(ob_space.shape)
        self.ac_space = ac_space.n
        self.encoder_scope = self.transition_scope = self.featurer_scope = self.decoder_scope = None
        self.default_encoder = self.default_transition = self.default_featurer = None
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.input_state = tf.placeholder(tf.float32, name="input_state",
                                          shape=[None] + self.ob_space)
        self.input_action = tf.placeholder(tf.int32, name="input_action",
                                           shape=[None])
        self.input_latent_state = tf.placeholder(tf.float32, name="input_latent_state",
                                                 shape=[None, self.latent_size])

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
                                 x.shape.as_list()))
            x = tf.reshape(x, new_shape)
            for i in range(3):
                x = tf.nn.elu(U.conv2d(x, 32, "conv%d"%i, filter_size=(3,3),
                                       stride=(2,2)))
            x = U.flattenallbut0(x)
            # x = tf.nn.relu(U.dense(x, 256, "dense1",
                                   # weight_init=U.normc_initializer()))
            output = U.dense(x, self.latent_size, "dense2",
                             weight_init=U.normc_initializer())
        return output

    def build_transition(self, latent_state, actions):
        '''
        Args:
            latent_state - n x latent_size latent encoding of state
            actions - n x t tensor encoding the action at each
                timestep
        Returns:
            next_states - an approximation of the latent encoding of the state
                for each future timestep given the application of the actions
        '''
        n_factors = 512
        lstm_model = True
        x = latent_state
        if not self.transition_scope:
            self.transition_scope = "transition"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.transition_scope, reuse=reuse) as scope:
            self.transition_scope = scope
            actions = tf.one_hot(actions, self.ac_space, axis=-1)
            if lstm_model:
                grua = GRUACell(self.latent_size, n_factors)
                self.state_init = x
                next_states, _ = tf.nn.dynamic_rnn(grua,
                                                   actions,
                                                   initial_state=self.state_init)
        return next_states

    def build_featurer(self, latent_state):
        if not self.featurer_scope:
            self.featurer_scope = "featurer"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.featurer_scope, reuse=reuse) as scope:
            self.featurer_scope = scope
            output = U.dense(latent_state, self.feature_size, "dense1",
                             weight_init=U.normc_initializer())
        return output

    # -------------- LOSS FUNCTIONS -----------------------------------

    def loss(self, states, actions, features, seq_length=None, x_to_f_ratio=1):
        '''
        Estimates MSE prediction loss for features given states, actions, and a
        true feature_extractor. REQUIRED: t >= 1
        Args:
            states - n x (t+1) x [self.ob_space] tensor
            actions - n x t tensor of action integers
            features - n x (t+1) x self.feature_size tensor
            seq_length - optional, tensor representing t
            x_to_f_ratio - (optional) a scalar weighting the latent vs feature
                loss. result_loss = feature_loss + x_to_f_ratio*latent_loss
        Returns:
            loss - MSE error for feature prediction across time
        '''
        summaries = []
        s0 = states[:, 0]
        x0 = self.build_encoder(s0)
        if seq_length is None:
            # TODO: assumes ob_space is 3 dimensional
            self.states = tf.slice(states, [0,0,0,0,0],
                                   tf.concat([-1, seq_length + 1, -1, -1, -1]))
            self.actions = tf.slice(actions, [0,0],
                                    tf.concat([-1, seq_length]))
            self.features = tf.slice(features, [0,0,0],
                                     tf.concat([-1, seq_length + 1, -1]))
        f_flattened = tf.reshape(features, [-1, self.feature_size])
        s_future = states[:, 1:]
        s_future_flattened = tf.reshape(s_future, [-1] + self.ob_space)
        x_future_flattened = self.build_encoder(s_future_flattened)
        # x_future = tf.reshape(x_future_flattened, [-1, t, self.latent_size])
        x_future_hat = self.build_transition(x0, actions)
        x_hat = tf.concat([tf.expand_dims(x0, axis=1), x_future_hat], axis=1)
        x_future_flattened_hat = tf.reshape(x_future_hat, [-1, self.latent_size])
        x_flattened_hat = tf.reshape(x_hat, [-1, self.latent_size])
        f_flattened_hat = self.build_featurer(x_flattened_hat)
        feature_loss = tf.reduce_mean(
            tf.squared_difference(f_flattened, f_flattened_hat),
            name="feature_loss")
        latent_loss = tf.reduce_mean(
            tf.squared_difference(x_future_flattened, x_future_flattened_hat),
            name='latent_loss')
        total_loss = tf.identity(feature_loss + x_to_f_ratio*latent_loss,
                                 name="overall_loss")

        summaries.extend([
            tf.summary.scalar('overall feature loss', feature_loss),
            tf.summary.scalar('overall latent loss', latent_loss),
            tf.summary.scalar('overall loss', total_loss),
            tf.summary.image('input', s0),
        ])

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        return total_loss, x_hat, var_list, tf.summary.merge(summaries)

    # -------------- UTILITIES ---------------------------------------

    def get_encoding(self, state):
        if self.default_encoder is None:
            self.default_encoder = self.build_encoder(self.input_states[0])
        single_run = state.ndim == 3 # TODO: avoid specifically naming # dim
        if single_run:
            state = np.expand_dims(state, 0)
        feed_dict = {self.input_state: state}
        latent_state = U.get_session().run(self.default_encoder, feed_dict)
        if single_run:
            latent_state = latent_state[0]
        return latent_state

    def get_future_encoding(self, latent_state, action):
        single_run = latent_state.ndim == 1
        if self.default_transition is None:
            self.default_transition = self.build_transition(
                self.input_latent_state,
                tf.expand_dims(self.input_action, axis=1))
        if single_run:
            latent_state = np.expand_dims(latent_state, 0)
            action = np.expand_dims(action, 0)
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_action: action}
        nx = U.get_session().run(self.default_transition, feed_dict=feed_dict)
        if single_run:
            nx = tf.squeeze(nx, 0)
        return nx

    def get_feature_from_encoding(self, latent_state):
        single_run = latent_state.ndim == 1
        if single_run:
            latent_state = np.expand_dims(latent_state, axis=0)
        if self.default_featurer is None:
            self.default_featurer = self.build_featurer(
                self.input_latent_state)
        if latent_state.ndim == 1:
            latent_state = np.expand_dims(latent_state, 0)
        feed_dict = {self.input_latent_state: latent_state}
        f = U.get_session().run(self.default_featurer, feed_dict=feed_dict)
        if single_run:
            f = np.squeeze(f, 0)
        return f
