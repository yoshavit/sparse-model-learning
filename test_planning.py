import argparse
import gym
import gym_mnist
import tensorflow as tf
import logging
from model import EnvModel
import configs
import numpy as np
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
from utils.latest_checkpoint_unsafe import latest_checkpoint
# restore_path = tf.train.latest_checkpoint(args.logdir)
restore_path = latest_checkpoint(args.logdir)
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
        states = np.zeros([0] + list(env.observation_space.shape))
        latents = np.zeros([0, config['latent_size']])
        for _ in range(4):
            arr = np.random.randint(10, size=[32])
            s = np.stack([coreenv._get_image_from_digit(i) for i in arr],
                         axis=0)
            x = envmodel.encode(s)
            states = np.concatenate([states, s], axis=0)
            latents = np.concatenate([latents, x], axis=0)
        states = np.stack(states, axis=0)
        latents = np.stack(latents, axis=0)
        lv.add_images(states, latents)

    cv2.namedWindow("env_state", cv2.WINDOW_NORMAL)
    cv2.namedWindow("env_goal", cv2.WINDOW_NORMAL)
    cv2.namedWindow("envmodel_state", cv2.WINDOW_NORMAL)
    for _ in range(30):
        s = env.reset()
        cv2.imshow("env_state", cv2.resize(s, None, fx=5, fy=5))
        x = envmodel.encode(s)
        done = False
        print("New game! Press q to quit")
        i = 0
        while not done:
            i += 1
            action = getch()
            if action == 'q':
                exit()
            elif action not in "wasd":
                print("Input must be WASD, was {}".format(action))
                continue # next loop iteration
            else:
                action = "wasd".find(action)
            s, rew, done, info = env.step(action)
            x = envmodel.stepforward(x, action)
            if use_latent_visualizer:
                x_visualized = lv.get_nearest_image(x)
                cv2.imshow("envmodel_state", cv2.resize(x_visualized, None,
                                                        fx=5, fy=5))
            gs = info['goal_state']
            gx = envmodel.encode(gs)
            rew_guess = envmodel.checkgoal(x, gx)
            cv2.imshow("env_state", cv2.resize(s, None,
                                               fx=5, fy=5))
            cv2.imshow("env_goal", cv2.resize(gs, None,
                                              fx=5, fy=5))
            print("Step {}, true goal: {}, estimated goal: {:0.3}".format(i, rew, rew_guess))
