import random
import numpy as np
import tensorflow as tf
from model import EnvModel
from utils.visualize_embeddings import images_to_sprite
from tensorflow.contrib.tensorboard.plugins import projector
from collections import deque
import scipy.misc
import os

REPLAY_MEMORY_SIZE = 500
DEFAULT_STEPSIZE = 1e-4


class ModelLearner:
    def __init__(self, env,
                 config,
                 summary_writer=None):
        """Creates a model-learner framework
        """
        self.replay_memory = []
        self.env = env
        self.config = config
        maxhorizon = config['maxhorizon']
        self.summary_writer=summary_writer
        feature_shape = config['feature_shape']
        self.global_step = tf.get_variable(
            "global_step", [], tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)
        self.states = tf.placeholder(tf.float32, shape=[None, maxhorizon + 1] +
                                      list(env.observation_space.shape), name="states")
        self.actions = tf.placeholder(tf.int32, shape=[None, maxhorizon], name="actions")
        self.goal_values = tf.placeholder(tf.float32, shape=[None,
                                                             maxhorizon],
                                          name="goal_values")
        self.goal_states = tf.placeholder(tf.float32, shape=[None] +
                                          list(env.observation_space.shape),
                                          name="goal_states")
        if config['feature_type'] == 'regression':
            self.features = tf.placeholder(tf.float32,
                                           shape=[None, maxhorizon+1] + feature_shape,
                                           name="features")
        elif config['feature_type'] == 'softmax':
            self.features = tf.placeholder(tf.int32,
                                           shape=[None,maxhorizon+1] + feature_shape[:-1],
                                           name="features")
        else:
            raise RuntimeError("Feature loss type must be either regression or softmax")
        self.seq_length = tf.placeholder(tf.int32, shape=[None, 1], name="seq_length")
        inc_step = self.global_step.assign_add(tf.shape(self.states)[0])
        self.local_steps = 0

        self.envmodel = EnvModel(env.observation_space,
                                 env.action_space,
                                 feature_shape,
                                 latent_size=config['latent_size'],
                                 transition_stacked_dim=config['transition_stacked_dim'],
                                 sigmoid_latents=config['sigmoid_latents'],
                                 feature_type=config['feature_type'])
        self.loss, latents, var_list, base_summary, timestep_summary = self.envmodel.loss(
            self.states,
            self.actions,
            self.features,
            goal_values=self.goal_values,
            goal_states=self.goal_states,
            seq_length=self.seq_length,
            max_horizon=config['maxhorizon'],
            x_to_f_ratio=config['x_to_f_ratio'],
            x_to_g_ratio=config['x_to_g_ratio'],
        )
        grads = tf.gradients(self.loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        grads_and_vars = list(zip(grads, var_list))
        self.train_op = tf.group(
            tf.train.AdamOptimizer(config['stepsize']).apply_gradients(grads_and_vars),
            inc_step)
        with tf.variable_scope("embedding"):
            self.num_embed_vectors = 1024
            latent_tensors = [tf.squeeze(tensor, axis=1)
                              for tensor in tf.split(latents,
                                                     maxhorizon+1,
                                                     axis=1)]
            self.latent_variables = [tf.Variable(
                tf.zeros([self.num_embed_vectors, config['latent_size']]),
                trainable=False, name="latent%i"%i)
                for i in range(maxhorizon + 1)]
            self.embeddings_op = [tf.assign(variable, tensor) for variable, tensor
                                  in zip(self.latent_variables, latent_tensors)]
            self.projectorconfig = projector.ProjectorConfig()
            logdir = self.summary_writer.get_logdir()
            for i in range(maxhorizon + 1):
                latent = self.latent_variables[i]
                embedding = self.projectorconfig.embeddings.add()
                embedding.tensor_name = latent.name
                embedding.sprite.image_path = os.path.join(logdir,
                                                           'embed_sprite%d.png'%i)
                if config['has_labels']:
                    embedding.metadata_path = os.path.join(logdir,
                                                           'embed_labels%d.tsv'%i)
        self.summary_op = tf.summary.merge([base_summary, timestep_summary])

    def train_model(self, sess, states, actions, features, goal_values,
                    goal_states, seq_lengths, labels=None,
                   show_embeddings=False):
        """
        Args:
            states - array of shape batch_size x T+1 x [state_shape]
            actions - array of shape batch_size x T x 1
            features - array batch_size x T+1 x feature_shape([:-1] in case of classification)
            goal_values
            goal_states
            seq_lengths - array batch_size x 1
            labels - (optional) array of batch_size x T+1 x 1
            sess - [MUST SPECIFY] current tensorflow session
            show_embeddings - if true, create embeddings
        """
        do_summary = self.local_steps%20 == 1
        if do_summary:
            fetches = [self.train_op, self.global_step, self.summary_op]
        else:
            fetches = [self.train_op, self.global_step]
        if show_embeddings:
            fetches += [self.embeddings_op]
        feed_dict = {self.seq_length: seq_lengths,
                     self.states: states,
                     self.actions: actions,
                     self.features: features,
                     self.goal_values: goal_values,
                     self.goal_states: goal_states,
                    }
        fetched = sess.run(fetches, feed_dict=feed_dict)
        self.local_steps += 1
        if do_summary:
            self.summary_writer.add_summary(fetched[2], fetched[1])
            self.summary_writer.flush()
        if show_embeddings:
            assert states.shape[0] == self.num_embed_vectors, "batch_size when embedding vectors must be modellearner.num_embed_vectors, because embedding variable size must be prespecified"
            for i in range(self.config['maxhorizon'] + 1):
                embedding = self.projectorconfig.embeddings[i]
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
                if self.config['has_labels']:
                    label_data = labels[:, i]
                    metadata_file = open(embedding.metadata_path, 'w')
                    metadata_file.write('Name\tClass\n')
                    for ll in range(label_data.shape[0]):
                        metadata_file.write('%06d\t%d\n' % (ll, label_data[ll]))
                    metadata_file.close()
            projector.visualize_embeddings(self.summary_writer, self.projectorconfig)

# ------------- UTILITIES ------------------------------

    def gather_gameplay_data(self, num_games, policy="random"):
        """Collects gameplay transition data and stores it in replay_memory

        Args:
            num_games: total number of games to simulate
            policy: a function mapping from an observation (or None on first
                timestep) to an action. Default is random.
        """
        if not callable(policy):
            policy = lambda obs, goal_obs: self.env.action_space.sample()

        for i in range(num_games):
            game_memory = []
            obs, goal_obs = self.env.reset()
            action_queue = deque()
            done = False
            c = 0
            while not done:
                c += 1
                if len(action_queue) == 0:
                    action_or_actions = policy(obs, goal_obs)
                    try:
                        action_queue.extend(action_or_actions)
                    except TypeError:
                        action_queue.append(action_or_actions)
                action = action_queue.popleft()
                new_obs, rew, done, info = self.env.step(action)
                game_memory.append((obs, action, new_obs, rew, bool(done), info))
                assert int(bool(rew)) == rew, "Game must have goal-type rewards"
                obs = new_obs
            self.replay_memory.append(tuple(game_memory))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory = self.replay_memory[-REPLAY_MEMORY_SIZE:]

    def create_transition_dataset(self, n=None, feature_from_info=True,
                                 variable_steps=True):
        """Constructs a list of model input matrices representing the
        components of the transitions. No guarantee of ordering or lack thereof ordering.

        Args:
            n - number of transitions to generate (if None use all transitions
                from games)
            feature_from_info - boolean, if true config['feature_extractor'] takes
                as input "info['state']" or "info['next_state']" as returned by
                the gym environment
        """
        assert self.replay_memory, "Must gather_gameplay_data before creating\
transition dataset!"
        transition_seqs = []
        max_steps = self.config['maxhorizon'] # num forward transition steps
        min_steps = self.config['minhorizon']
        candidate_steps = list(range(min_steps, max_steps+1)) if variable_steps else [max_steps]
        for game in self.replay_memory:
            for n_steps in candidate_steps:
                for i in range(len(game) - n_steps + 1):
                    transitions = game[i:i+n_steps]
                    # compile states, then pad with zeros
                    states = [transitions[j][0] for j in range(n_steps)] +\
                            [transitions[-1][2]] +\
                            [np.zeros_like(transitions[0][0])
                             for j in range(max_steps - n_steps)]
                    actions = [transitions[j][1] for j in range(n_steps)] +\
                            [np.zeros_like(transitions[0][1])
                             for j in range(max_steps - n_steps)]
                    goal_values = [transitions[j][3] for j in range(n_steps)] +\
                            [np.zeros_like(transitions[0][1])
                             for j in range(max_steps - n_steps)]
                    goal_states = transitions[0][5]['goal_state']
                    feature_extractor = self.config['feature_extractor']
                    if feature_from_info:
                        features = [feature_extractor(transitions[j][5]['state'])
                                    for j in range(n_steps)] +\
                                [feature_extractor(transitions[-1][5]['next_state'])]
                    else:
                        features = [self.feature_extractor(s) for s in states]
                    features += [np.zeros_like(features[0])
                                 for j in range(max_steps - n_steps)]
                    if self.config['has_labels']:
                        label_extractor = self.config['label_extractor']
                        labels = [label_extractor(
                            transitions[j][5]['state']) for j in range(n_steps)] +\
                            [label_extractor(transitions[-1][5]['next_state'])]
                        labels += [np.zeros_like(labels[0]) for j in
                                   range(max_steps - n_steps)]
                    seq_length = [n_steps]
                    transition_seq = [states, actions, features, goal_values,
                                      goal_states, seq_length]
                    if self.config['has_labels']:
                        transition_seq.append(labels)
                    transition_seqs.append(transition_seq)
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
        # counts = [0]*10
        # for i in range(len(output[-1])):
            # for j in range(len(output[-1][0])):
                # counts[output[-1][i][j][0]] += 1
        return output

# TODO: feature_extractor specs are inconsistent - some take single val, others
# multiple
