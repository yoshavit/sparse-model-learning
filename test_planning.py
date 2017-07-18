import argparse
import gym
import gym_mnist
import tensorflow as tf
import logging
from model import EnvModel
import configs
import numpy as np
import random
from utils.latent_visualizer import LatentVisualizer
from utils.getch import getch
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Tests the planning module of an env model")
parser.add_argument('logdir', help="Path to weights directory")
args = parser.parse_args()
logdir = args.logdir
config = configs.load_config(logdir)
env = gym.make(config['env'])
envmodel = EnvModel(env.observation_space,
                    env.action_space,
                    config['feature_shape'],
                    latent_size=config['latent_size'],
                    transition_stacked_dim=config['transition_stacked_dim'],
                    feature_type=config['feature_type'])
restore_path = tf.train.latest_checkpoint(args.logdir)
# from tensorflow.python import pywrap_tensorflow
# reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
    # print("tensor_name: ", key)
var_dict = {var.name[:-2]: var for var in tf.global_variables()}

restoring_saver = tf.train.Saver(var_list=var_dict)
with tf.Session() as sess:
    if restore_path is not None:
        logger.info("Restoring variables from checkpoint: {}".format(restore_path))
        restoring_saver.restore(sess, restore_path)
    else:
        logger.info("Initializing brand new network parameters.")
        sess.run(tf.global_variables_initializer())


    # Prepare a group of images (3 from each class, 10 classes)
    coreenv = env.unwrapped
    # NOTE: only valid for basic mnist environments
    use_latent_visualizer = isinstance(coreenv, gym_mnist.envs.MNISTEnv)
    if use_latent_visualizer:
        lv = LatentVisualizer()
        states = []
        latents = []
        for _ in range(100):
            i = random.randint(10)
            s = coreenv._get_image_from_digit(i)
            x = envmodel.encode(s)
            states.append(s); latents.append(x)
        states = np.stack(states, axis=0)
        latents = np.stack(latents, axis=0)
        lv.add_images(states, latents)

    cv2.namedWindow("env_state", cv2.WINDOW_NORMAL)
    cv2.namedWindow("env_goal", cv2.WINDOW_NORMAL)
    cv2.namedWindow("envmodel_state", cv2.WINDOW_NORMAL)
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
            if use_latent_visualizer:
                x_visualized = lv.get_nearest_image(x)
                cv2.imshow("envmodel_state", x_visualized)
            gs = info['goal_state']
            gx = envmodel.encode(gs)
            rew_guess = envmodel.checkgoal(x, gx)
            cv2.imshow("env_state", s)
            cv2.imshow("env_goal", gs)
            print("True goal: {}, estimated goal: {}".format(rew, rew_guess))
            cv2.waitKey(1000)
