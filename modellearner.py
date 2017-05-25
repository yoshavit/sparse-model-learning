import random
import numpy as np
import tensorflow as tf
from model import EnvModel
from utils.visualize_embeddings import images_to_sprite
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import os

REPLAY_MEMORY_SIZE = 1000
DEFAULT_STEPSIZE = 1e-4

"""
"""

class ModelLearner:
    def __init__(self, env, feature_size,
                 stepsize=DEFAULT_STEPSIZE,
                 latent_size=10,
                 max_horizon=3,
                 summary_writer=None,
                 feature_extractor=None):
        """Creates a model-learner framework

        Args:
            max_horizon - number of steps transition model rolls forward (1 means
            1-step lookahead)
            feature_extractor - (optional) takes as input an array of states (concated
                along 0 dimension), outputs an array of features
        """
        self.replay_memory = []
        self.env = env
        self.feature_extractor = feature_extractor
        self.max_horizon = max_horizon
        self.summary_writer=summary_writer
        with tf.variable_scope("envmodel"):
            self.envmodel = EnvModel(env.observation_space, env.action_space,
                                     feature_size, latent_size)
        self.global_step = tf.get_variable("global_step", [], tf.int32,
                                           initializer=tf.constant_initializer(0, dtype=tf.int32),
                                           trainable=False)
        self.states = [tf.placeholder(tf.float32, shape=[None] +
                                      list(env.observation_space.shape),
                                      name="state%d"%i) for i in
                       range(max_horizon + 1)]
        self.actions = [tf.placeholder(tf.int32, shape=[None],
                                       name="action%d"%i) for i in
                        range(max_horizon)]
        self.features = [tf.placeholder(tf.float32, shape=[None, feature_size],
                                        name="feature%d"%i) for i in
                         range(max_horizon + 1)]
        self.seq_length = tf.placeholder(tf.int32, shape=[], name="seq_length")
        inc_step = self.global_step.assign_add(tf.shape(self.states)[0])
        self.local_steps = 0

        with tf.variable_scope("train"):
            self.loss, latents, var_list, summary = self.envmodel.loss(
                tf.stack(self.states, axis=1),
                tf.stack(self.actions, axis=1),
                tf.stack(self.features, axis=1),
                seq_length=self.seq_length,
                x_to_f_ratio=1)
            grads = tf.gradients(self.loss, var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_and_vars = list(zip(grads, var_list))
            self.train_op = tf.group(
                tf.train.AdamOptimizer(stepsize).apply_gradients(grads_and_vars),
                inc_step)
        with tf.variable_scope("embedding"):
            self.num_embed_vectors = 256
            latent_tensors = [tf.squeeze(tensor, axis=1)
                              for tensor in tf.split(latents,
                                                     self.max_horizon+1,
                                                     axis=1)]
            self.latent_variables = [tf.Variable(tf.zeros([self.num_embed_vectors, latent_size]),
                                                 trainable=False,
                                                 name="latent%i"%i)
                                     for i in range(max_horizon + 1)]
            self.embeddings_op = [tf.assign(variable, tensor) for variable, tensor
                                  in zip(self.latent_variables, latent_tensors)]
            self.config = projector.ProjectorConfig()
            logdir = self.summary_writer.get_logdir()
            for i in range(max_horizon + 1):
                latent = self.latent_variables[i]
                embedding = self.config.embeddings.add()
                embedding.tensor_name = latent.name
                embedding.sprite.image_path = os.path.join(logdir,
                                                           'embed_sprite%d.png'%i)
                embedding.metadata_path = os.path.join(logdir,
                                                       'embed_labels%i.tsv'%i)
        self.summary_op = tf.summary.merge_all()

    def train_model(self, states, actions, features, sess,
                   show_embeddings=False):
        """
        Args:
            states - array of shape batch_size x t+1 x [state_shape]
            actions - array of shape batch_size x t x 1
            features - array batch_size x t+1 x self.feature_size
            sess - current tensorflow session
            show_embeddings - if true, create embeddings
        """
        T = actions.shape[1]
        do_summary = self.local_steps%20 == 1
        if do_summary:
            fetches = [self.train_op, self.global_step, self.summary_op]
        else:
            fetches = [self.train_op, self.global_step]
        if show_embeddings:
            fetches += [self.embeddings_op]
        feed_dict = {self.seq_length: T,
                     self.states[0]: states[:, 0],
                     self.features[0]: features[:, 0]}
        for t in range(T):
            feed_dict[self.states[t+1]] = states[:, t+1]
            feed_dict[self.actions[t]] = actions[:, t]
            feed_dict[self.features[t+1]] = features[:, t+1]
        fetched = sess.run(fetches, feed_dict=feed_dict)
        self.local_steps += 1
        if do_summary:
            self.summary_writer.add_summary(fetched[2], fetched[1])
            self.summary_writer.flush()
        if show_embeddings:
            assert states.shape[0] == self.num_embed_vectors
            for i in range(self.max_horizon + 1):
                embedding = self.config.embeddings[i]
                # images
                image_data = states[:, i]
                thumbnail_size = image_data.shape[1]
                if len(embedding.sprite.single_image_dim) == 0:
                    embedding.sprite.single_image_dim.extend(
                        [thumbnail_size, thumbnail_size])
                sprite = images_to_sprite(image_data)
                if sprite.shape[2] == 1:
                    sprite = sprite[:,:,0]
                scipy.misc.imsave(embedding.sprite.image_path, sprite)
                # labels
                label_data = features[:, i]
                # TODO: allow for custom labels, rather than just features
                assert label_data.shape[1] == 1
                metadata_file = open(embedding.metadata_path, 'w')
                metadata_file.write('Name\tClass\n')
                for ll in range(label_data.shape[0]):
                    metadata_file.write('%06d\t%d\n' % (ll, label_data[ll]))
                metadata_file.close()
            projector.visualize_embeddings(self.summary_writer, self.config)

# ------------- UTILITIES ------------------------------

    def gather_gameplay_data(self, num_games, policy="random"):
        """Collects gameplay transition data and stores it in replay_memory

        Args:
            num_games: total number of games to simulate
            policy: a function mapping from an observation (or None on first
                timestep) to an action. Default is random.
        """
        if not callable(policy):
            policy = lambda obs: self.env.action_space.sample()

        for i in range(num_games):
            game_memory = []
            obs = self.env.reset()
            done = False
            c = 0
            while not done:
                c += 1
                action = policy(obs)
                new_obs, rew, done, info = self.env.step(action)
                game_memory.append((obs, action, new_obs, rew, bool(done), info))
                obs = new_obs
            self.replay_memory.append(game_memory)
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory = self.replay_memory[-REPLAY_MEMORY_SIZE:]

    def create_transition_dataset(self, steps, n=None, feature_from_info=True):
        """Constructs a list of model input matrices representing the
        components of the transitions. No guarantee of ordering or lack thereof ordering.

        Args:
            steps - number of forward transition steps
            n - number of transitions to generate (if None use all transitions
                from games)
            feature_from_info - boolean, if true self.feature_extractor takes
                as input "info['state']" or "info['next_state']" as returned by
                the gym environment
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
                if feature_from_info:
                    features = [self.feature_extractor(
                        transitions[j][5]['state']) for j in range(steps)] +\
                        [self.feature_extractor(transitions[-1][5]['next_state'])]
                else:
                    features = [self.feature_extractor(s) for s in states]
                transition_seqs.append([states, actions, features])
        if n and n < len(transition_seqs):
            transition_seqs = random.sample(transition_seqs, n)
        # convert to list of arrays
        output = []
        for i in range(len(transition_seqs[0])):
            output.append([])
            for j in range(len(transition_seqs[0][i])):
                output[i].append(np.asarray([transition_seqs[k][i][j]
                                          for k in
                                          range(len(transition_seqs))]))
            output[i] = np.stack(output[i], axis=1)
        return output

    # def visualize_embeddings(self, sess, logdir, n=1000):
        # states, actions, features = self.create_transition_dataset(self.max_horizon, n=n)
        # transition_data = [states[:, i] for i in range(states.shape[1])] +\
                # [actions[:, i] for i in range(actions.shape[1])]
        # if features.shape[2] == 1 and features.dtype == int:
            # labels = [features[:, i] for i in range(features.shape[1])]
        # else:
            # raise NotImplementedError("Can only use 1-d int features as labels")
        # vis_mapping = range(states.shape[1])
        # input_placeholders = self.states + self.actions
        # visualize_embeddings(os.path.join(logdir, 'embeddings'),
                             # self.latent_tensors,
                             # sess,
                             # transition_data,
                             # vis_mapping=vis_mapping,
                             # labels=labels,
                             # input_placeholders=input_placeholders,
                             # summary_writer=self.summary_writer)

        # Alternative implementation:
            # tf.FIFOQueue(100, [tf.float32]*t + [tf.float32]*t)


# TODO: feature_extractor specs are inconsistent - some take single val, others
# multiple
