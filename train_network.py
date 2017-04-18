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
from modellearner import *
from utils.visualize_embeddings import visualize_embeddings
from utils.save_and_load import save_scope, load_scope, get_scope_vars

def load_classic_mnist_data(mnist_env):
    data, labels = mnist_env._get_full_mnist_dataset()
    return data, labels

def increment_path(origpath):
    subscript = None
    newpath = origpath
    while True:
        if os.path.exists(newpath):
            if subscript is None:
                newpath = origpath + "--00"
                subscript = 0
            else:
                subscript += 1
                newpath = "%s--%.2i"%(origpath, subscript)
        else:
            break
    return newpath

parser = argparse.ArgumentParser(description="Trains a model_learner on the MNIST game.")
parser.add_argument('game', default="mnist-v0", help="Name of environment to be trained on. Examples: 'mnist-v0', 'blockwalker-v0', and 'blockwalker-multicolored-v0'")
parser.add_argument('--name', default="default", help="Name of current experiment.")
parser.add_argument('--load', help="Path to a folder containing the checkpoint for a network with the same architecture, relative to file")
parser.add_argument('--project', action="store_true",
                    help="whether to generate a latent-space projection")
parser.add_argument('--train', action="store_true", help="Train the model")
parser.add_argument('-batchsize', help="Batch size for learning", type=int,
                    default=32)
parser.add_argument('--stationary', action="store_true",
                    help="Whether to use a fixed or varying game experience distribution for training")
parser.add_argument('--visualize', action="store_true",
                    help="Display a few sample predictions after each epoch")
parser.add_argument('-epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('-stepsize', type=float, default=1e-3, help="train step size")
parser.add_argument('-latentdims', type=int, default=10, help="Number of latent dimensions used to encode the state")

args = parser.parse_args()
# --------- TRAINER ARGUMENTS ----------------------------------------
CUSTOM_NAME = args.name
if args.load:
    reuse_weights = True
    OLD_LOG_DIR = args.load
else:
    reuse_weights = False
game = args.game
training = args.train
projecting = args.project
stationary_distribution = args.stationary
visualize = args.visualize
num_epochs = args.epochs
batch_size = args.batchsize
stepsize = args.stepsize
latentdims=args.latentdims
summarize = True
no_encoder_gradient = False
initial_num_games = 1000
num_games = 100
n = 10000
trained_pair = [0, 1]
max_horizon = 1
# ------------ CREATE MODEL TRAINER -----------------------------
BASE_DIR = os.path.join("./data", game, CUSTOM_NAME)
LOG_DIR = increment_path(os.path.join(BASE_DIR, "logdir"))
print("Writing results to " + LOG_DIR)
mnist_env = gym.make(game)
ml = ModelLearner(mnist_env.observation_space,
                  mnist_env.action_space,
                  max_horizon,
                  latent_dims=latentdims,
                  no_encoder_gradient=no_encoder_gradient,
                  residual=True)
update_steps, losses = ml.build_pair_cluster_trainer(
    trained_pair, pos_weight_multiplier=2, stepsize=stepsize)
# u, l = ml.build_reward_trainer(stepsize=stepsize)
# update_steps += u; losses += l
loss_names = [loss.name for loss in losses]
test_input_placeholders, test_output_placeholders, get_distance, get_reward =\
    ml.build_pair_cluster_executer(1)
ml.gather_gameplay_data(mnist_env, initial_num_games)
transition_seqs = ml.create_true_false_transition_dataset(
    n, fraction_true=1/3, fraction_from_same_game=0.7)
init = tf.global_variables_initializer()
performupdate = U.function(ml.get_full_inputs(),
                           losses, update_steps)
saved_scopes = [ml.encoder_scope, ml.transition_scope]

with tf.Session() as sess:
    sw = tf.summary.FileWriter(LOG_DIR, sess.graph)
    init.run()
    if reuse_weights:
        for sc in saved_scopes:
            load_scope(
                os.path.join(OLD_LOG_DIR, sc.name + "_weights.npz"), sc)
        print("Loaded previous weights")
    if training:
        for i in range(num_epochs):
            print ("Epoch %d: "%i)
            if not stationary_distribution and i%3==0:
                ml.gather_gameplay_data(mnist_env, num_games)
                transition_seqs = ml.create_true_false_transition_dataset(
                    n, fraction_true=1/3, fraction_from_same_game=0.7)
            batch_losses = np.zeros(len(losses))
            l = 0
            for batch in tqdm(dataset.iterbatches(transition_seqs,
                                      batch_size=batch_size, shuffle=True)):
                loss_results = performupdate(*batch)
                # if l%5 == 0:
                    # loss_results = performupdate_with_mse(*batch)
                # else:
                    # loss_results = performupdate(*batch)
                batch_losses += np.array(loss_results)
                l +=1
            batch_losses = batch_losses/l
            for name, loss in zip(loss_names, batch_losses):
                print(name + ": %.5f"%loss)
            if visualize:
                t_inds = random.sample(range(n), 4)
                t = [np.array([transition_seqs[i][j] for j in t_inds])
                              for i in range(len(test_input_placeholders))]
                l = np.array([transition_seqs[-1][j] for j in t_inds])
                plt.clf()
                ml.visualize_transitions(
                    t, labels=l, get_distance=get_distance,
                    get_reward=get_reward,
                    action_names=mnist_env.get_action_meanings())
                plt.pause(2)
            for sc in saved_scopes:
                save_scope(
                    os.path.join(LOG_DIR, sc.name + "_weights.npz"), sc)
    if visualize and not training:
        for x in range(num_epochs):
            t_inds = random.sample(range(n), 4)
            t = [np.array([transition_seqs[i][j] for j in t_inds])
                          for i in range(len(test_input_placeholders))]
            l = np.array([transition_seqs[-1][j] for j in t_inds])
            plt.clf()
            ml.visualize_transitions(
                t, labels=l, get_distance=get_distance,
                get_reward=get_reward,
                action_names=mnist_env.get_action_meanings())
            plt.pause(5)
    if projecting:
        # IF TRAIN AND TEST FORMATS DIVERGE, CHANGE TO USE TRAIN DATASET
        visualize_embeddings(
            LOG_DIR,
            test_output_placeholders,
            sess,
            transition_seqs[:-2],
            summary_writer=sw,
            labels=transition_seqs[-1],
            vis_mapping=[1, 1],
            data_placeholders=test_input_placeholders
        )
    sw.close()
    print("To visualize results, call:")
    print('tensorboard --logdir=%s' % LOG_DIR)

