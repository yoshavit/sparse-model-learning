import numpy as np
import tensorflow as tf
from utils import tf_util as U
from utils.gruacell import GRUACell

class EnvModel:
    def __init__(self, ob_space, ac_space, feature_size, max_horizon=10, latent_size=128):
        """Creates a model-learner framework
        max_horizon is the # of unrolled steps, with 0 unrolled steps meaning
        max_horizon = 0
        """
        self.ob_space = list(ob_space.shape)
        self.ac_space = ac_space.n
        self.encoder_scope = self.transition_scope = self.featurer_scope = None
        self.default_encoder = self.default_transition = self.default_featurer = None
        self.max_horizon = max_horizon
        self.latent_size = latent_size
        self.feature_size = feature_size

        # CURRENTLY NOT REALLY USED:
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

    def build_transition(self, latent_state, actions, seq_length=None):
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
                # gru = tf.contrib.rnn.GRUCell(self.latent_size)
                self.state_init = x
                next_states, _ = tf.nn.dynamic_rnn(grua,
                                                   actions,
                                                   initial_state=self.state_init,
                                                   sequence_length=seq_length)
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
        true feature_extractor. REQUIRED: T = self.max_horizon >= t > 0
        Note that each vector should contain t (or t+1) elements, and the rest
        should be zeros (up to self.max_horizon)
        Args:
            states - n x (T+1) x [self.ob_space] tensor (up to t+1 unzeroed)
            actions - n x T tensor of action integers (up to t unzeroed)
            features - n x (T+1) x self.feature_size tensor (up to t+1 unzeroed)
            seq_length - optional, n x 1 tensor representing t's,
            x_to_f_ratio - (optional) a scalar weighting the latent vs feature
                loss. result_loss = feature_loss + x_to_f_ratio*latent_loss
        Returns:
            loss - MSE error for feature prediction across time
        '''
        summaries = []
        T = tf.shape(actions)[1]
        s0 = states[:, 0]
        x0 = self.build_encoder(s0)
        f = features
        seq_length = tf.squeeze(seq_length, axis=1)
        if seq_length is None:
            seq_length = tf.reduce_sum(tf.ones_like(actions), axis=1)


        def zero_out(inp, startindex, totallength, axis=1):
            # zero out the elements of input starting at start_index along axis
            # startindex is equivalent to seq_length
            with tf.variable_scope("zero_out"):
                crange = tf.range(totallength, dtype=tf.int32)
                for i in range(len(inp.shape)):
                    # expand validrange's dimensions
                    if i < axis:
                        crange = tf.expand_dims(crange, 0)
                    elif i == axis: # the axis itself is already expanded
                        pass
                    else: # i > axis
                        crange = tf.expand_dims(crange, -1)
                    # expand start_index's dimensions
                    if i == 0:
                        pass
                    else:
                        startindex = tf.expand_dims(startindex, -1)
                input_shape = tf.shape(inp)
                crange_tileshape = tf.concat([
                    input_shape[:axis], [1], input_shape[axis+1:]], axis=0)
                startindex_tileshape = tf.concat([[1], input_shape[1:]], axis=0)
                crange_mesh = tf.tile(crange, crange_tileshape)
                startindex_mesh = tf.tile(startindex, startindex_tileshape)
                validmesh = tf.cast(tf.less(crange_mesh, startindex_mesh),
                                    tf.float32)
                output = tf.multiply(inp, validmesh)
            return output

        # Create containers of size 0 for states, actions, features
        # if seq_length is not None:
            # states = tf.slice(states, [0,0,0,0,0],
                                   # tf.stack([-1, seq_length + 1, -1, -1, -1]))
            # actions = tf.slice(actions, [0,0],
                                    # tf.stack([-1, seq_length]))
            # features = tf.slice(features, [0,0,0],
                                     # tf.stack([-1, seq_length + 1, -1]))

        # def pad_to_len(inp, length):
            # # pad input's axis=1 to length with zeros
            # with tf.variable_scope("pad_to_maxlen"):
                # pad_shape = np.zeros([len(input.shape), 2])
                # pad_shape[1,1] = 1
                # pad_shape = tf.constant(pad_shape, dtype=tf.int32)*\
                    # (length - tf.shape(inp)[1])
            # return tf.pad(inp, pad_shape, "CONSTANT")

        s_future = states[:, 1:]
        s_future_flattened = tf.reshape(s_future, [-1] + self.ob_space)
        x_future_flattened = self.build_encoder(s_future_flattened)
        # zero-out unnecessary elements from x_future_flattened
        x_future = tf.reshape(x_future_flattened, [-1, T, self.latent_size])
        # x_future = tf.reshape(x_future_flattened, [-1, t, self.latent_size])
        x_future_hat = self.build_transition(x0, actions, seq_length)
        x_hat = tf.concat([tf.expand_dims(x0, axis=1), x_future_hat], axis=1)
        x_flattened_hat = tf.reshape(x_hat, [-1, self.latent_size])
        f_flattened_hat = self.build_featurer(x_flattened_hat)
        f_hat = tf.reshape(f_flattened_hat, [-1, T+1, self.feature_size])

        feature_diff = zero_out(tf.squared_difference(f, f_hat), seq_length+1,
                                self.max_horizon + 1)
        feature_loss = tf.reduce_mean(feature_diff,
            name="feature_loss")
        latent_diff = zero_out(tf.squared_difference(x_future, x_future_hat),
                               seq_length, self.max_horizon)
        latent_loss = tf.reduce_mean(latent_diff,
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

    def encode(self, state):
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

    def advance_encoding(self, latent_state, action):
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

    def feature_from_encoding(self, latent_state):
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
