import gym
import gym_mnist
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import argparse
from utils import dataset, tf_util as U
from utils.latent_visualizer import LatentVisualizer
from utils.visualize_embeddings import do_elementwise_eval
from modellearner import *
from utils.save_and_load import save_scope, load_scope, get_scope_vars
"""
Create image database by exploring the game
Visualize closest 1-step transition from a number of different frames
If planning:
    pick a start state
    do bfs until you reach a terminal=1 target
    return the sequence of actions, and simulate what executing those actions
        would look like, compare the simulation to visualizing each of the
        latent frames
"""
parser = argparse.ArgumentParser(description="Trains a model_learner on the MNIST game.")
parser.add_argument('game', help="Name of environment to be trained on. Examples: 'mnist-v0', 'blockwalker-v0', and 'blockwalker-multicolored-v0'")
parser.add_argument('filename', help="Path to a folder containing the saved game data relative to file")
parser.add_argument('-latentdims', type=int, default=32, help="Number of latent dimensions used to encode the state")

args = parser.parse_args()
# --------- TRAINER ARGUMENTS ----------------------------------------
filename = args.filename
game = args.game
latentdims = args.latentdims
summarize = True
num_games = 100
epochs = 30
max_horizon = 2
# ------------ CREATE MODEL TRAINER -----------------------------
LOG_DIR = os.path.join(filename, "testing")
IMAGES_DIR = os.path.join(LOG_DIR, "images")
print("Writing results to " + LOG_DIR)
env = gym.make(game)
ml = ModelLearner(env.observation_space,
                  env.action_space,
                  max_horizon,
                  latent_dims=latentdims,
                  residual=True)
test_input_placeholders, test_output_placeholders, get_distance =\
    ml.build_pair_cluster_executer(1)
lv = LatentVisualizer()
init = tf.global_variables_initializer()
saved_scopes = [ml.encoder_scope, ml.transition_scope]
with tf.Session() as sess:
    sw = tf.summary.FileWriter(LOG_DIR, sess.graph)
    init.run()
    for sc in saved_scopes:
        load_scope(
            os.path.join(filename, sc.name + "_weights.npz"), sc)
    print("Loaded previous weights")
    if tf.gfile.Exists(IMAGES_DIR):
        print("Loading old gameplay images...")
        lv.load(IMAGES_DIR)
        game_images = lv.images
        game_latents = lv.latents
        game_images_minus_terminals = [np.array(i) for i in
                                       game_images.tolist()]
        # TODO: actually reconstruct without terminals
    else:
        game_images = []
        game_images_minus_terminals = []
        print("Creating new gameplay images...")
        ml.gather_gameplay_data(env, num_games)
        for game in ml.replay_memory:
            for i in range(len(game)):
                game_images.append(game[i][0])
                game_images_minus_terminals.append(game[i][0])
            game_images.append(game[-1][2])
        for i in game_images:
            print(i.shape)
        game_images = np.stack(game_images)
        s0 = ml.input_states[0]
        x0 = ml.build_encoder(s0)
        game_latents = do_elementwise_eval(x0, s0, game_images)
        lv.add_images(game_images, game_latents)
        print("Saving newly generated images.")
        tf.gfile.MakeDirs(IMAGES_DIR)
        lv.save(IMAGES_DIR)
    for i in range(epochs):
        n = 5
        _, axarr = plt.subplots(n, 2, num=1)
        plt.subplots_adjust(hspace=1)
        action_names = env.get_action_meanings()
        print ("Finding %d approximate transitions"%n)
        states = random.sample(game_images_minus_terminals, n)
        latents = [ml.get_encoding(s) for s in states]
        actions = [ml.ac_space.sample() for i in range(n)]
        next_latents = [ml.get_transition_from_encoding(x, a)[0] for x, a in
                        zip(latents, actions)]
        guessed_next_states = [lv.get_nearest_image(nl) for nl in next_latents]
        for i in range(n):
            plt.sca(axarr[i, 0])
            plt.axis("off")
            ac_string = ""
            ac_seq = [actions[i]]
            before = states[i]
            after = guessed_next_states[i]
            for ac in ac_seq:
                if action_names:
                    ac_string += ", " + action_names[ac]
                else:
                    ac_string += ", " + str(ac)
            txt = "Actions: " + ac_string
            plt.text(35, 0, txt) # adjust for appropriate spacing
            plt.imshow(before, cmap="Greys")
            plt.title("Before")
            plt.sca(axarr[i, 1])
            plt.axis("off")
            plt.imshow(after, cmap="Greys")
            plt.title("After")
            plt.pause(0.01)
        plt.pause(10)
        plt.clf()

    sw.close()

