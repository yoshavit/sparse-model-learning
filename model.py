import numpy as np
import tensorflow as tf
import os
from utils import tf_util as U

class EnvModel:
    def __init__(self, ob_space, ac_space, feature_shape,
                 latent_size=128, transition_stacked_dim=1):
        """Creates a model-learner framework
        transition_stacked_dim is the number of stacked cells for each timestep
        in the transition model
        """
        self.ob_space = list(ob_space.shape)
        self.ac_space = ac_space.n
        self.encoder_scope = self.transition_scope = self.featurer_scope = self.goaler_scope = None
        self.default_encoder = self.default_transition = self.default_featurer = self.default_goaler = None
        self.latent_size = latent_size
        self.feature_shape = feature_shape
        self.transition_stacked_dim = transition_stacked_dim

        self.test_batchsize = 32
        self.input_state = tf.placeholder(tf.float32, name="input_state",
                                          shape=[self.test_batchsize] + self.ob_space)
        self.input_actions = tf.placeholder(tf.int32, name="input_actions",
                                           shape=[self.test_batchsize, None])
        self.input_latent_state = tf.placeholder(tf.float32, name="input_latent_state",
                                                 shape=[self.test_batchsize, self.latent_size])
        self.input_latent_goalstate = tf.placeholder(tf.float32,
                                                     name="input_latent_goalstate",
                                              shape=[self.test_batchsize] + self.ob_space)

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
                x = tf.nn.elu(U.conv2d(x, 32, "conv%d"%i, filter_size=(5,5),
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
        # n_factors = 512
        x = latent_state
        if not self.transition_scope:
            self.transition_scope = "transition"
            reuse = False
        else:
            reuse = True

        with tf.variable_scope(self.transition_scope, reuse=reuse) as scope:
            self.transition_scope = scope
            actions = tf.one_hot(actions, self.ac_space, axis=-1)
            # def cell():
                # return GRUACell(
                # self.latent_size//self.transition_stacked_dim, n_factors)
            def cell():
                return tf.contrib.rnn.GRUBlockCell(
                    self.latent_size)
            # note that we use a special version of MultiRNNCell
            # to expose the entire RNN's internal state at each timestep
            # stacked_cell = ExposedMultiRNNCell(
                # [cell() for _ in range(self.transition_stacked_dim)])
            stacked_cell = tf.contrib.rnn.MultiRNNCell(
                [cell() for _ in range(self.transition_stacked_dim)])
            initial_state = tuple([x] + [tf.zeros_like(x) for _ in range(self.transition_stacked_dim - 1)])
            next_states, _ = tf.nn.dynamic_rnn(stacked_cell,
                                               actions,
                                               initial_state=initial_state,
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
            feature_size = np.prod(self.feature_shape)
            output = U.dense(latent_state, feature_size, "dense1",
                             weight_init=U.normc_initializer())
            output = tf.reshape(output, [-1] + self.feature_shape)
        return output

    def build_goaler(self, latent_state, latent_goal_state=None):
        if not self.goaler_scope:
            self.goaler_scope = "goaler"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.goaler_scope, reuse=reuse) as scope:
            self.goaler_scope = scope
            if latent_goal_state is not None:
                x = tf.concat([latent_state, latent_goal_state], axis=1)
            else:
                x = latent_state
            x = U.dense(x, 256, "dense1",
                        weight_init=U.normc_initializer())
            x = U.dense(x, 1, "dense2",
                        weight_init=U.normc_initializer())
            output = tf.squeeze(x, axis=1)
        return output

    # -------------- LOSS FUNCTIONS -----------------------------------

    def loss(self,
             states,
             actions,
             features,
             goal_states=None,
             goal_values=None,
             seq_length=None,
             max_horizon=10,
             x_to_f_ratio=1,
             x_to_g_ratio=1,
             feature_regression=False,
             feature_softmax=False):
        '''
        Estimates MSE prediction loss for features given states, actions, and a
        true feature_extractor. REQUIRED: T = self.max_horizon >= t > 0
        Note that each vector should contain t (or t+1) elements, and the rest
        should be zeros (up to self.max_horizon)
        Args:
            states - n x (T+1) x [self.ob_space] tensor (up to t+1 unzeroed)
            actions - n x T tensor of action integers (up to t unzeroed)
            features - n x (T+1) x self.feature_shape tensor (up to t+1 unzeroed)
            goal_states - n x [self.ob_space]
            goal_values - n x T tensor (1 or 0) representing whether goal
                was reached after t transitions
            seq_length - optional, n x 1 tensor representing t's,
            x_to_f_ratio - (optional) a scalar weighting the latent vs feature
                loss. result_loss = feature_loss + x_to_f_ratio*latent_loss
        Returns:
            loss - MSE error for embedding prediction across time, plus feature
                loss (either regression or softmax classificaiton loss)
        '''
        summaries = []
        T = actions.get_shape().as_list()[1]
        use_goals = goal_values is not None
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

        s_future = states[:, 1:]
        s_future_flattened = tf.reshape(s_future, [-1] + self.ob_space)
        x_future_flattened = self.build_encoder(s_future_flattened)
        x_future = tf.reshape(x_future_flattened, [-1, T, self.latent_size])
        x_future_hat = self.build_transition(x0, actions, seq_length)
        x_future_flattened_hat = tf.reshape(x_future_hat, [-1, self.latent_size])
        x_hat = tf.concat([tf.expand_dims(x0, axis=1), x_future_hat], axis=1)
        x_flattened_hat = tf.reshape(x_hat, [-1, self.latent_size])
        f_flattened_hat = self.build_featurer(x_flattened_hat)
        f_hat = tf.reshape(f_flattened_hat, [-1, T+1] + self.feature_shape)

        if use_goals:
            g = goal_values
            if goal_states is None:
                g_hat = self.build_goaler(x_future_flattened_hat)
            else:
                sg = goal_states
                xg = self.build_encoder(sg)
                xg_padded = tf.tile(xg, [T, 1])
                g_hat = self.build_goaler(x_future_flattened_hat, xg_padded)
            g_hat = tf.reshape(g_hat, [-1, T])
            goal_diff = zero_out(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=g,
                                                        logits=g_hat),
                seq_length, max_horizon)
            goal_loss = tf.reduce_mean(goal_diff, name="goal_loss")

        if feature_regression:
            feature_diff = zero_out(tf.squared_difference(f, f_hat), seq_length+1,
                                    max_horizon + 1)
        elif feature_softmax:
            feature_diff = zero_out(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_hat,
                                                               labels=f),
                seq_length + 1, max_horizon + 1)
        else:
            raise RuntimeError("Feature loss must be either regression or softmax")

        feature_loss = tf.reduce_mean(feature_diff, name="feature_loss")
        latent_diff = zero_out(tf.squared_difference(x_future, x_future_hat),
                               seq_length, max_horizon)
        latent_loss = tf.reduce_mean(latent_diff, name='latent_loss')
        if use_goals:
            total_loss = tf.identity(feature_loss + x_to_f_ratio*latent_loss +
                                     x_to_g_ratio*goal_loss,
                                     name="overall_loss")
        else:
            total_loss = tf.identity(feature_loss + x_to_f_ratio*latent_loss,
                                     name="overall_loss")

        summaries.extend([
            tf.summary.scalar('overall feature loss', feature_loss),
            tf.summary.scalar('overall latent loss', latent_loss),
            tf.summary.scalar('overall loss', total_loss),
            tf.summary.image('input', s0),
        ])
        if use_goals: summaries.append(tf.summary.scalar('overall goal loss',
                                                          goal_loss))

        #timestep summaries
        with tf.variable_scope("timestep"):
            # feature loss in each timestep, averaged over each nonzero feature
            feature_losses = [
                tf.div(
                    tf.reduce_sum(tf.squeeze(t, axis=1), axis=0),
                    tf.count_nonzero(tf.squeeze(t, axis=1), axis=0,
                                     dtype=tf.float32) + 1e-12, # to avoid dividing by 0
                    name="feature_loss%i"%i)
                for i, t in enumerate(
                    tf.split(tf.reduce_mean(feature_diff,
                                            axis=list(range(2, len(f.shape)))),
                             max_horizon+1, axis=1))]
            # same, but latent state loss
            latent_losses = [
                tf.div(
                    tf.reduce_sum(tf.squeeze(t, axis=1), axis=0),
                    tf.count_nonzero(tf.squeeze(t, axis=1), axis=0,
                                     dtype=tf.float32) + 1e-12,
                    name="latent_loss%i"%(i+1))
                for i, t in enumerate(
                    tf.split(tf.reduce_mean(latent_diff, axis=2),
                             max_horizon, axis=1))]
            # same, but for goals
            if use_goals:
                goal_losses = [
                    tf.div(
                        tf.reduce_sum(tf.squeeze(t, axis=1), axis=0),
                        tf.count_nonzero(tf.squeeze(t, axis=1), axis=0,
                                         dtype=tf.float32) + 1e-12,
                        name="goals_loss%i"%(i+1))
                    for i, t in enumerate(
                        tf.split(goal_diff,
                                 max_horizon, axis=1))]
            else:
                goal_losses = []
            timestep_summaries = []
            for loss in feature_losses + latent_losses + goal_losses:
                _, loss_name = os.path.split(loss.name)
                timestep_summaries.extend([
                    tf.summary.scalar(loss_name, loss),
                    tf.summary.histogram(loss_name, loss)
                ])
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        return total_loss, x_hat, var_list, tf.summary.merge(summaries),\
                tf.summary.merge(timestep_summaries)

    # -------------- UTILITIES ---------------------------------------

    def encode(self, state):
        if self.default_encoder is None:
            self.default_encoder = self.build_encoder(self.input_states[0])
        single_run = state.ndim == len(self.ob_space)
        if single_run:
            nstate = np.zeros([self.test_batchsize] + self.ob_space)
            nstate[0] += state
            state = nstate
        feed_dict = {self.input_state: state}
        latent_state = U.get_session().run(self.default_encoder, feed_dict)
        if single_run:
            latent_state = latent_state[0]
        return latent_state

    def stepforward(self, latent_state, actions):
        single_run = latent_state.ndim == 1
        actions = np.asarray(actions)
        if actions.ndim == 0: actions = np.expand_dims(actions, 1)
        if self.default_transition is None:
            self.default_transition = self.build_transition(
                self.input_latent_state,
                self.input_actions)
        if single_run:
            nlatent_state = np.zeros([self.test_batchsize, self.latent_size])
            nlatent_state[0] += latent_state
            latent_state = nlatent_state
            nactions = np.zeros([self.test_batchsize, actions.shape[0]])
            nactions[0] += actions
            actions = nactions
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_action: actions}
        nxs = U.get_session().run(self.default_transition, feed_dict=feed_dict)
        nx = nxs[:, -1]
        if single_run:
            nx = nx[0]
        return nx

    def feature_from_encoding(self, latent_state):
        single_run = latent_state.ndim == 1
        if single_run:
            nlatent_state = np.zeros([self.test_batchsize, self.latent_size])
            nlatent_state[0] += latent_state
            latent_state = nlatent_state
        if self.default_featurer is None:
            self.default_featurer = self.build_featurer(
                self.input_latent_state)
        feed_dict = {self.input_latent_state: latent_state}
        f = U.get_session().run(self.default_featurer, feed_dict=feed_dict)
        if single_run:
            f = f[0]
        return f

    def checkgoal(self, latent_state, latent_goal=None):
        single_run = latent_state.ndim == 1
        if single_run:
            nlatent_state = np.zeros([self.test_batchsize, self.latent_size])
            nlatent_state[0] += latent_state
            latent_state = nlatent_state
        if self.default_goaler is None:
            self.default_goaler = self.build_goaler(latent_state, latent_goal)
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_goalstate: latent_goal}
        g = U.get_session().run(self.default_goaler, feed_dict=feed_dict)
        if single_run:
            g = g[0]
        return g
