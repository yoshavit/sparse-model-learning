import os
import argparse
import gym
import gym_mnist
import tensorflow as tf
import logging
from model import EnvModel
from utils.latent_visualizer import LatentVisualizer
from utils.getch import getch
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Trains a model_learner on the MNIST game.")
parser.add_argument('env', default="mnist-v0", help="Name of environment to"
                    "be trained on. Examples: 'mnist-v0', 'blockwalker-v0',"
                    "and 'blockwalker-multicolored-v0'")
parser.add_argument('weightpath', help="Path to weights directory")

args = parser.parse_args()
# ----------------------------------------------
"""
Things needed in config file:
"""
# simple mnist config
mnist_config = {
    'env': 'mnist-v0',
    'stepsize': 1e-4,
    'maxsteps': 10000000,
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1],
    'feature_regression': True,
    'feature_softmax': False,
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': 64,
    'maxhorizon': 5,
    'force_latent_consistency': True,
    'transition_stacked_dim': 1,
    'minhorizon': 1,
    'n_initial_games': 300,
    'use_goalstates': True,
}
# simple multi-goal config
mnist_multigoal_config = {
    'env': 'mnist-multigoal-v0',
    'stepsize': 1e-4,
    'maxsteps': 10000000,
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1],
    'feature_regression': True,
    'feature_softmax': False,
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': 64,
    'maxhorizon': 5,
    'force_latent_consistency': True,
    'transition_stacked_dim': 1,
    'minhorizon': 1,
    'n_initial_games': 500,
    'use_goalstates': True,
}
config = mnist_config
assert args.env == config['env']

env = gym.make(config['env'])
weightdir = args.weightpath
envmodel = EnvModel(env.observation_space,
                    env.action_space,
                    config['feature_shape'],
                    latent_size=config['latent_size'],
                    transition_stacked_dim=config['transition_stacked_dim'])
restore_path = tf.train.latest_checkpoint(weightdir)
# from tensorflow.python import pywrap_tensorflow
# reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
    # print("tensor_name: ", key)
var_dict = {"train/"+var.name[:-2]: var for var in tf.global_variables()}

restoring_saver = tf.train.Saver(var_list=var_dict)
with tf.Session() as sess:
    if restore_path is not None:
        logger.info("Restoring variables from checkpoint: {}".format(restore_path))
        restoring_saver.restore(sess, restore_path)
    else:
        logger.info("Initializing brand new network parameters.")
        sess.run(tf.global_variables_initializer())
    cv2.namedWindow("env", cv2.WINDOW_NORMAL)
    cv2.namedWindow("envmodel", cv2.WINDOW_NORMAL)
    for _ in range(30):
        s = env.reset()
        cv2.imshow("env", s)
        x = envmodel.encode(s)
        done = False
        print("New game! Press q to quit")
        while not done:
            action = getch()
            if action == 'q':
                exit()
            elif action not in "wasd":
                print("Input must be WASD, was {}".format(action))
                continue # next loop iteration
            else:
                action = "wasd".find(action)
                print("Action: {}".format(action))
            s, rew, done, info = env.step(action)
            x = envmodel.stepforward(x, action)
            gs = info['goal_state']
            gx = envmodel.encode(gs)
            rew_guess = envmodel.checkgoal(x, gx)
            cv2.imshow("env", s)
            print("True goal: {}, estimated goal: {}".format(rew, rew_guess))
    # lv = LatentVisualizer()

    # if tf.gfile.Exists(IMAGES_DIR):
        # print("Loading old gameplay images...")
        # lv.load(IMAGES_DIR)
        # game_images = lv.images
        # game_latents = lv.latents
        # game_images_minus_terminals = [np.array(i) for i in
                                       # game_images.tolist()]
        # # TODO: actually reconstruct without terminals
    # else:
        # game_images = []
        # game_images_minus_terminals = []
        # print("Creating new gameplay images...")
        # ml.gather_gameplay_data(env, num_games)
        # for game in ml.replay_memory:
            # for i in range(len(game)):
                # game_images.append(game[i][0])
                # game_images_minus_terminals.append(game[i][0])
            # game_images.append(game[-1][2])
        # for i in game_images:
            # print(i.shape)
        # game_images = np.stack(game_images)
        # s0 = ml.input_states[0]
        # x0 = ml.build_encoder(s0)
        # game_latents = do_elementwise_eval(x0, s0, game_images)
        # lv.add_images(game_images, game_latents)
        # print("Saving newly generated images.")
        # tf.gfile.MakeDirs(IMAGES_DIR)
        # lv.save(IMAGES_DIR)
    # for i in range(epochs):
        # n = 5
        # _, axarr = plt.subplots(n, 2, num=1)
        # plt.subplots_adjust(hspace=1)
        # action_names = env.get_action_meanings()
        # print ("Finding %d approximate transitions"%n)
        # states = random.sample(game_images_minus_terminals, n)
        # latents = [ml.get_encoding(s) for s in states]
        # actions = [ml.ac_space.sample() for i in range(n)]
        # next_latents = [ml.get_transition_from_encoding(x, a)[0] for x, a in
                        # zip(latents, actions)]
        # guessed_next_states = [lv.get_nearest_image(nl) for nl in next_latents]
        # for i in range(n):
            # plt.sca(axarr[i, 0])
            # plt.axis("off")
            # ac_string = ""
            # ac_seq = [actions[i]]
            # before = states[i]
            # after = guessed_next_states[i]
            # for ac in ac_seq:
                # if action_names:
                    # ac_string += ", " + action_names[ac]
                # else:
                    # ac_string += ", " + str(ac)
            # txt = "Actions: " + ac_string
            # plt.text(35, 0, txt) # adjust for appropriate spacing
            # plt.imshow(before, cmap="Greys")
            # plt.title("Before")
            # plt.sca(axarr[i, 1])
            # plt.axis("off")
            # plt.imshow(after, cmap="Greys")
            # plt.title("After")
            # plt.pause(0.01)
        # plt.pause(10)
        # plt.clf()

