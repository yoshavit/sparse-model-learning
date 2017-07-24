import numpy as np
import tensorflow as tf
from utils import tf_util as U

class EnvModel:
    def __init__(self, ob_space, ac_space, feature_shape,
                 uses_goal_states=True,
                 latent_size=128, transition_stacked_dim=1,
                 feature_type="regression", sigmoid_latents=True):
        """Creates a model-learner framework
        transition_stacked_dim is the number of stacked cells for each timestep
        in the transition model
        """
        self.ob_space = list(ob_space.shape)
        self.ac_space = ac_space.n
        self.encoder_scope = self.transition_scope = self.featurer_scope = self.goaler_scope = None
        self.latent_size = latent_size
        self.sigmoid_latents = sigmoid_latents # pass latents through sigmoid
        self.feature_shape = feature_shape
        self.feature_type = feature_type
        assert self.feature_type in ["regression", "softmax"], "Feature loss must be either regression or softmax, was {}".format(self.feature_type)
        self.transition_stacked_dim = transition_stacked_dim

        self.test_batchsize = 32
        self.input_state = tf.placeholder(tf.float32, name="input_state",
                                          shape=[self.test_batchsize] + self.ob_space)
        self.input_actions = tf.placeholder(tf.int32, name="input_actions",
                                            shape=[self.test_batchsize, None])
        self.input_latent_state = tf.placeholder(tf.float32, name="input_latent_state",
                                                 shape=[self.test_batchsize, self.latent_size])
        if uses_goal_states:
            self.input_latent_goalstate = tf.placeholder(
                tf.float32, name="input_latent_goalstate",
                shape=[self.test_batchsize, self.latent_size])
        else:
            self.input_latent_goalstate = None
        self.default_encoder = self.build_encoder(self.input_state,
                                                  reuse=False)
        self.default_transition = self.build_transition(self.input_latent_state,
                                                        self.input_actions,
                                                        reuse=False)
        self.default_goaler, _ = self.build_goaler(self.input_latent_state,
                                                self.input_latent_goalstate,
                                                reuse=False)
        self.default_featurer, _ = self.build_featurer(self.input_latent_state,
                                                    reuse=False)

#------------------------ MODEL SUBCOMPONENTS ----------------------------------

    def build_encoder(self, input, reuse=True):
        x = input
        with tf.variable_scope("encoder", reuse=reuse) as scope:
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
            output_logits = U.dense(x, self.latent_size, "dense2",
                             weight_init=U.normc_initializer())
            output = output_logits
        return output

    def build_transition(self, latent_state, actions, seq_length=None,
                         reuse=True):
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
        with tf.variable_scope("transition", reuse=reuse) as scope:
            self.transition_scope = scope
            actions = tf.one_hot(actions, self.ac_space, axis=-1)
            # def cell():
                # return GRUACell(
                # self.latent_size//self.transition_stacked_dim, n_factors)
            # if self.sigmoid_latents:
                # class SigmoidGRUCell(tf.nn.contrib.rnn.GRUBlockCell):
                    # def __call__(self, x, h_prev, scope=None):
                        # new_h, _ = super(SigmoidGRUCell, self).__call__(x, h_prev, scope)
                        # from tensorflow.python.ops import variable_scope as vs
                        # with vs.variable_scope(scope or type(self).__name__):
                            # new_h_sigmoided = tf.nn.sigmoid(new_h)
                            # return new_h, new_h_sigmoided
                # def cell():
                    # return SigmoidGRUCell(self.latent_size)
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

    def build_featurer(self, latent_state, reuse=True):
        with tf.variable_scope("featurer", reuse=reuse) as scope:
            self.featurer_scope = scope
            feature_size = np.prod(self.feature_shape)
            if self.sigmoid_latents:
                x = tf.sigmoid(latent_state)
            else:
                x = latent_state
            x = tf.nn.elu(U.dense(
                x, 256, "dense1",
                weight_init=U.normc_initializer()))
            x = U.dense(x, feature_size, "dense2",
                        weight_init=U.normc_initializer())
            output_logits = tf.reshape(x, [-1] + self.feature_shape)
            if self.feature_type == "regression":
                output = output_logits
            elif self.feature_type == "softmax":
                output = tf.nn.softmax(output_logits)
        return output, output_logits

    def build_goaler(self, latent_state, latent_goal_state=None, reuse=True):
        with tf.variable_scope("goaler", reuse=reuse) as scope:
            self.goaler_scope = scope
            if latent_goal_state is not None:
                x = tf.concat([latent_state, latent_goal_state], axis=1)
            else:
                x = latent_state
            if self.sigmoid_latents:
                x = tf.sigmoid(x)
            x = tf.nn.elu(
                U.dense(x, 256, "dense1", weight_init=U.normc_initializer()))
            x = U.dense(x, 1, "dense2",
                        weight_init=U.normc_initializer())
            x = tf.squeeze(x, axis=1)
            output_logits = x
            output = tf.sigmoid(output_logits)
        return output, output_logits

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
             x_to_g_ratio=1):
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
        _, f_flattened_hat_logits = self.build_featurer(x_flattened_hat)
        f_hat_logits = tf.reshape(f_flattened_hat_logits, [-1, T+1] +
                                  self.feature_shape)
        if use_goals:
            g = goal_values
            if goal_states is None:
                _, g_hat_logits = self.build_goaler(x_future_flattened_hat)
            else:
                sg = goal_states
                xg = self.build_encoder(sg)
                xg = tf.expand_dims(xg, axis=1)
                xg_padded = tf.tile(xg, [1, T, 1])
                xg_flattened = tf.reshape(xg_padded, [-1, self.latent_size])
                _, g_hat_logits = self.build_goaler(x_future_flattened_hat,
                                                    xg_flattened)
            g_hat_logits = tf.reshape(g_hat_logits, [-1, T])
            goal_diff = zero_out(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=g,
                                                        logits=g_hat_logits),
                seq_length, max_horizon)
            goal_loss = tf.reduce_mean(goal_diff, name="goal_loss")

        if self.feature_type == "regression":
            feature_diff = zero_out(tf.squared_difference(f, f_hat_logits), seq_length+1,
                                    max_horizon + 1)
        elif self.feature_type == "softmax":
            feature_diff = zero_out(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_hat_logits,
                                                               labels=f),
                seq_length + 1, max_horizon + 1)
        else:
            raise RuntimeError()

        feature_loss = tf.reduce_mean(feature_diff, name="feature_loss")

        if self.sigmoid_latents:
            latent_diff = zero_out(tf.norm(tf.sigmoid(x_future) -
                                           tf.sigmoid(x_future_hat),
                                           ord=1, axis=2),
                                   seq_length, max_horizon)
        else:
            latent_diff = tf.reduce_mean(
                zero_out(tf.squared_difference(x_future, x_future_hat),
                         seq_length, max_horizon),
                axis=2)
        latent_loss = tf.reduce_mean(latent_diff, name='latent_loss')
        if use_goals:
            total_loss = tf.identity(latent_loss + x_to_f_ratio*feature_loss +
                                     x_to_g_ratio*goal_loss,
                                     name="overall_loss")
        else:
            total_loss = tf.identity(feature_loss + x_to_f_ratio*latent_loss,
                                     name="overall_loss")

        with tf.variable_scope("overall"):
            summaries.extend([
                tf.summary.scalar('feature loss', feature_loss),
                tf.summary.scalar('latent loss', latent_loss),
                tf.summary.scalar('loss', total_loss),
                tf.summary.image('input', s0),
            ])
            if use_goals: summaries.append(tf.summary.scalar('goal loss',
                                                              goal_loss))

        #timestep summaries

        timestep_summaries = []
        with tf.variable_scope("features"):
            # feature loss in each timestep, averaged over each nonzero feature
            feature_losses = [
                tf.div(
                    tf.reduce_sum(tf.squeeze(t, axis=1), axis=0),
                    tf.count_nonzero(tf.squeeze(t, axis=1), axis=0,
                                     dtype=tf.float32) + 1e-12 # to avoid dividing by 0
                    )
                for t in tf.split(tf.reduce_mean(feature_diff,
                                            axis=list(range(2, len(f.shape)))),
                             max_horizon+1, axis=1)]
            timestep_summaries.extend([tf.summary.scalar(
                "loss%i"%i, t) for i, t in enumerate(feature_losses)])

        with tf.variable_scope("latents"):
            # same, but latent state loss
            latent_losses = [
                tf.div(
                    tf.reduce_sum(tf.squeeze(t, axis=1), axis=0),
                    tf.count_nonzero(tf.squeeze(t, axis=1), axis=0,
                                     dtype=tf.float32) + 1e-12,
                    name="loss%i"%(i+1))
                for i, t in enumerate(
                    tf.split(latent_diff,
                             max_horizon, axis=1))]
            timestep_summaries.extend([tf.summary.scalar(
                "loss%i"%i, t) for i, t in enumerate(latent_losses)])
        # same, but for goals
        with tf.variable_scope("goals"):
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
            timestep_summaries.extend([tf.summary.scalar(
                "loss%i"%i, t) for i, t in enumerate(goal_losses)])
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        if self.sigmoid_latents:
            latent_results = tf.sigmoid(x_hat)
        else:
            latent_results = x_hat
        return total_loss, latent_results, var_list, tf.summary.merge(summaries),\
                tf.summary.merge(timestep_summaries)

    # -------------- UTILITIES ---------------------------------------

    def encode(self, state):
        single_run = state.ndim == len(self.ob_space)
        batchsize = 1 if single_run else state.shape[0]
        state = self.__reshape_batch(state, self.ob_space)
        feed_dict = {self.input_state: state}
        latent_state = U.get_session().run(self.default_encoder, feed_dict)
        latent_state = latent_state[:batchsize]
        if single_run: latent_state = latent_state[0]
        return latent_state

    def stepforward(self, latent_state, actions):
        single_run = latent_state.ndim == 1
        batchsize = 1 if single_run else latent_state.shape[0]
        actions = np.asarray(actions)
        if actions.ndim == 0: actions = np.expand_dims(actions, 1)
        actions_dim_minus_batchdim = actions.shape[0] if single_run else actions.shape[1]
        latent_state = self.__reshape_batch(latent_state, [self.latent_size])
        actions = self.__reshape_batch(actions,
                                       [actions_dim_minus_batchdim])
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_actions: actions}
        nxs = U.get_session().run(self.default_transition, feed_dict=feed_dict)
        nxs = nxs[:batchsize]
        nx = nxs[:, -1]
        if single_run: nxs = nxs[0]; nx = nx[0]
        return nx, nxs

    def getfeatures(self, latent_state):
        single_run = latent_state.ndim == 1
        batchsize = 1 if single_run else latent_state.shape[0]
        latent_state = self.__reshape_batch(latent_state, [self.latent_size])
        feed_dict = {self.input_latent_state: latent_state}
        f = U.get_session().run(self.default_featurer, feed_dict=feed_dict)
        f = f[:batchsize]
        if single_run: f = f[0]
        return f

    def checkgoal(self, latent_state, latent_goal=None):
        single_run = latent_state.ndim == 1
        batchsize = 1 if single_run else latent_state.shape[0]
        latent_state = self.__reshape_batch(latent_state, [self.latent_size])
        latent_goal = self.__reshape_batch(latent_goal, [self.latent_size])
        feed_dict = {self.input_latent_state: latent_state,
                     self.input_latent_goalstate: latent_goal}
        g = U.get_session().run(self.default_goaler, feed_dict=feed_dict)
        g = g[:batchsize]
        if single_run: g = g[0]
        return g

    def __reshape_batch(self, inp, dims_minus_batchdim):
        if inp.ndim == len(dims_minus_batchdim):
            inp = np.expand_dims(inp, 0)
        if inp.shape[0] < self.test_batchsize:
            tmp = inp
            inp = np.zeros([self.test_batchsize] + dims_minus_batchdim)
            inp[:tmp.shape[0]] = tmp[:]
            return inp
        elif inp.shape[0] == self.test_batchsize:
            return inp
        else:
            raise ValueError(
                "Input's batchsize must be smaller than"
                "envmodel.test_batchsize; was {} > {}".format(inp.shape[0],
                                                              self.test_batchsize))



