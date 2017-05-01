import numpy as np
import tensorflow as tf
import random
from utils import tf_util as U
from modellearner import ModelLearner

class FeatureBasedModelLearner(ModelLearner):
    def __init__(self, ob_space, ac_space, max_horizon,
                 feature_extractor,
                 feature_size,
                 latent_dims=10,
                 residual=True,
                 recurrent=False):
        """
        feature_extractor takes as input a state, and outputs a featureset
        """
        super().__init__(ob_space, ac_space, max_horizon,
                                           latent_dims=latent_dims,
                                           residual=residual)
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        self.featurer_scope = None
        if recurrent:
            raise NotImplementedError
        self.build_additional_inputs()

    def build_additional_inputs(self):
        self.input_features = []
        for t in range(self.max_horizon + 1):
            self.input_features.append(U.get_placeholder("input_feature_%d"%t,
                                                         tf.float32,
                                                         [None,
                                                          self.feature_size]))

    def get_full_inputs(self):
        return self.input_states + self.input_actions + self.input_rewards +\
                self.input_features

    def create_transition_dataset(self, steps, n=None, feature_from_info=True):
        """Constructs a list of model input matrices representing the
        components of the transitions.

        Args:
            steps - number of forward transition steps
            n - number of transitions to generate (if None use all transitions
                from games)
            feature_from_info - boolean, if true self.feature_extractor takes
                as input "info['state']" or "info['next_state']" as returned by
                the gym environment

        Output:
            A list of arrays, in the order:
                s_0 +
                [s_i, i = 1 to steps] +
                f_0 +
                [f_i, i = 1 to steps] +
                [a_i-1, i = 1 to steps] +
                [r_i-1, i = 1 to steps]
        """
        assert self.replay_memory, "Must gather_gameplay_data before creating\
transition dataset!"
        transition_seqs = []
        for game in self.replay_memory:
            for i in range(len(game) - steps + 1):
                transitions = game[i:i+steps]
                states = [transitions[j][0] for j in range(steps)] +\
                    [transitions[-1][2]]
                actions = [transitions[j][1] for j in range(steps)]
                rewards = [transitions[j][3] for j in range(steps)]
                # terminals = [transitions[j][4] for j in
                             # range(steps)]
                if feature_from_info:
                    features = [self.feature_extractor(
                        transitions[j][5]['state']) for j in range(steps)] +\
                        [self.feature_extractor(transitions[-1][5]['next_state'])]
                else:
                    features = [self.feature_extractor(s) for s in states]
                transition_seqs.append([states, actions, rewards, features])
        if n and n < len(transition_seqs):
            transition_seqs = random.sample(transition_seqs, n)
        random.shuffle(transition_seqs)
        # convert to list of arrays
        output = []
        for i in range(len(transition_seqs[0])):
            for j in range(len(transition_seqs[0][i])):
                output.append(np.asarray([transition_seqs[k][i][j]
                                          for k in
                                          range(len(transition_seqs))]))
        return output

    def build_featurer(self, x):
        if not self.featurer_scope:
            self.featurer_scope = "featurer"
            reuse = False
        else:
            reuse = True
        with tf.variable_scope(self.featurer_scope, reuse=reuse) as scope:
            self.featurer_scope = scope
            output = U.dense(x, self.feature_size, "dense1",
                             weight_init=U.normc_initializer())
        return output

    def get_feature_from_encoding(self, latent_state):
        if self.default_featurer is None:
            self.default_featurer = self.build_featurer(
                self.input_latent_state)
        if latent_state.ndim == 1:
            latent_state = np.expand_dims(latent_state, 0)
        feed_dict = {self.input_latent_state: latent_state}
        f = U.get_session().run(self.default_featurer, feed_dict=feed_dict)
        return np.squeeze(f, 0)

    def build_feature_reward_loss_trainer(self,
                                          forward_steps,
                                          with_reward=True,
                                          only_final_step=False,
                                          stepsize=1e-4,
                                          name=None,
                                          feature_scalar=1,
                                          reward_scalar=1):
        losses = []
        reward_loss_total = 0
        state_prediction_loss = [0 for t in range(forward_steps)]
        feature_prediction_loss = [0 for t in range(forward_steps + 1)]
        s0 = self.input_states[0]
        x0 = self.build_encoder(s0)
        f0 = self.input_features[0]
        f0_hat = self.build_featurer(x0)
        feature_prediction_loss[0] = tf.reduce_mean(
            tf.squared_difference(f0, f0_hat),
            name="0_step_feature_prediction_loss")
        x_hat = x0
        for t in range(1, forward_steps + 1):
            a = self.input_actions[t-1]
            x_hat, r_hat, _ = self.build_transition(x_hat, a)
            f_hat = self.build_featurer(x_hat)
            s = self.input_states[t]
            x = self.build_encoder(s)
            r = self.input_rewards[t-1]
            f = self.input_features[t]
            state_prediction_loss[t-1] = tf.reduce_mean(
                tf.squared_difference(x, x_hat),
                name="%d_step_state_prediction_loss"%t)
            feature_prediction_loss[t] = tf.reduce_mean(
                feature_scalar*tf.squared_difference(f, f_hat),
                name="%d_step_feature_prediction_loss"%t)
            if with_reward and (not only_final_step or t == forward_steps):
                reward_loss_total += tf.reduce_mean(tf.squared_difference(r,
                                                                          r_hat))
        if with_reward:
            reward_loss_total = tf.identity(reward_scalar*reward_loss_total,
                                            name="reward_loss")
        if only_final_step:
            state_prediction_loss = [state_prediction_loss[-1]]
            feature_prediction_loss = [feature_prediction_loss[0],
                                       feature_prediction_loss[-1]]
        cumloss = tf.add_n(state_prediction_loss + feature_prediction_loss +
                           [reward_loss_total], name="cum_loss")
        losses.append(reward_loss_total)
        losses.extend(state_prediction_loss)
        losses.extend(feature_prediction_loss)
        losses.append(cumloss)
        update_step = tf.train.AdamOptimizer(stepsize).minimize(cumloss)
        return [update_step], losses
