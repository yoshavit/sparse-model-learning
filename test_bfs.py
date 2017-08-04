import argparse
import gym
import gym_mnist
import tensorflow as tf
import logging
from model import EnvModel
import configs
import numpy as np
import cv2
import agents
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Tests the planning module of an env model")
parser.add_argument('logdir', help="Path to weights directory")
args = parser.parse_args()
logdir = args.logdir
config = configs.load_config(logdir)
env = gym.make(config['env'])
from utils.latest_checkpoint_unsafe import latest_checkpoint
restore_path = latest_checkpoint(args.logdir)
envmodel = EnvModel(env.observation_space,
                    env.action_space,
                    config['feature_shape'],
                    latent_size=config['latent_size'],
                    transition_stacked_dim=config['transition_stacked_dim'],
                    feature_type=config['feature_type'])
var_dict = {var.name[:-2]: var for var in tf.global_variables()}
restoring_saver = tf.train.Saver(var_list=var_dict)
with tf.Session() as sess:
    if restore_path is not None:
        logger.info("Restoring variables from checkpoint: {}".format(restore_path))
        restoring_saver.restore(sess, restore_path)
    else:
        logger.info("Initializing brand new network parameters.")
        sess.run(tf.global_variables_initializer())

    agent = agents.BFSAgent(envmodel, horizon=5)
    cv2.namedWindow("env_state", cv2.WINDOW_NORMAL)
    for _ in range(30):
        s, gs = env.reset()
        cv2.imshow("env_state", cv2.resize(s, None, fx=5, fy=5))
        cv2.waitKey(0)
        actionq = collections.deque()
        done = False
        i = 0
        while not done:
            i += 1
            if len(actionq) == 0:
                if i != 1:
                    print("Queue empty! Plan didn't work, replanning.")
                actions, final_goal_likelihood = agent.policy(s, gs)
                print(actions)
                print("New plan has probability {:0.3} of working.".format(final_goal_likelihood))
                try:
                    actionq.extend(actions)
                except TypeError:
                    actionq.append(actions)
            action = actionq.popleft()
            s, rew, done, info = env.step(action)
            print("Executed action: {}".format(env.unwrapped.get_action_meanings()[action]))
            gs = info['goal_state']
            cv2.imshow("env_state", cv2.resize(s, None, fx=5, fy=5))
            cv2.waitKey(0)
