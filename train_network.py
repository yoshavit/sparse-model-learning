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

def train_classic_mnist():
    CUSTOM_NAME = "ideal_encoder"
    BASE_DIR = os.path.join("./data", CUSTOM_NAME)
    LOG_DIR = increment_path(os.path.join(BASE_DIR, "logdir"))
    training = True
    reuse_weights = False
    projecting = True
    num_epochs = 1
    batch_size = 64
    if reuse_weights:
        custom_weight_dir = os.path.join(BASE_DIR, "logdir")
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                                 mnist_env.action_space, latent_dims=10)
    ml.build_mnist_classifier_model(stepsize=1e-3)
    digit_data, digit_labels = load_classic_mnist_data(mnist_env)
    pairs = list(zip(digit_data, digit_labels))[:10000]
    data, labels = zip(*pairs)
    init = tf.global_variables_initializer()

    placeholders = [ml.digit_input, ml.digit_labels]
    losses = [ml.loss, ml.classification_rate]
    performupdate = U.function(placeholders, losses +
                               [ml.train_step])

    print("Training an encoder on the original MNIST dataset.")
    with tf.Session() as sess:
        init.run()
        sw = tf.summary.FileWriter(LOG_DIR, sess.graph)
        if reuse_weights:
            saver.restore(sess, custom_weight_dir)
            print("Loaded previous weights")
        if training:
            for i in range(num_epochs):
                print ("Epoch %d: "%i)
                batch_losses = np.zeros(len(losses))
                l=0
                for batch in tqdm(dataset.iterbatches([data, labels],
                                                      batch_size=batch_size,
                                                      shuffle=True)):
                    *losses, _ = performupdate(*batch)
                    batch_losses += np.array(losses)
                    l += 1
                batch_loss, batch_classification_rate = batch_losses/l
                print ("Log loss: ~%0.5f, training accuracy: ~%0.5f" %(
                    batch_loss, batch_classification_rate))
        if projecting:
            visualize_embeddings(LOG_DIR, [ml.logits],
                                          sess, [data], labels=labels,
                                          data_placeholders=[ml.digit_input],
                                          labels_placeholder=ml.digit_labels,
                                          )
        sw.close()
        print("To visualize results, call:")
        print('tensorboard --logdir=%s' % LOG_DIR)

def __main__():
    parser = argparse.ArgumentParser(description="Trains a model_learner on the MNIST game.")
    parser.add_argument('--name', default="default", help="Name of current experiment.")
    parser.add_argument('--load',
                        help="Path to a folder containing the checkpoint for a network with the same\
architecture, relative to file")
    parser.add_argument('--load-encoder',
                        help="Path to a data file for the encoder only")
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

    args = parser.parse_args()
    # --------- TRAINER ARGUMENTS ----------------------------------------
    CUSTOM_NAME = args.name
    if args.load:
        reuse_weights = True
        OLD_LOG_DIR = args.load
    else:
        reuse_weights = False
    training = args.train
    projecting = args.project
    stationary_distribution = args.stationary
    visualize = args.visualize
    num_epochs = args.epochs
    batch_size = args.batchsize
    stepsize = args.stepsize
    summarize = True
    no_encoder_gradient = False
    num_games = 1000
    n = 5000
    trained_pair = [0, 1]
    max_horizon = 1
    # ------------ CREATE MODEL TRAINER -----------------------------
    BASE_DIR = os.path.join("./data", CUSTOM_NAME)
    LOG_DIR = increment_path(os.path.join(BASE_DIR, "logdir"))
    print("Writing results to " + LOG_DIR)
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                      mnist_env.action_space,
                      max_horizon,
                      latent_dims=10,
                      no_encoder_gradient=no_encoder_gradient,
                      residual=True)
    update_steps, losses = ml.build_pair_classifier_trainer(
        trained_pair, use_mse_loss=True,
        pos_weight_multiplier=3, stepsize=stepsize)
    loss_names = [loss.name for loss in losses]
    test_placeholders, get_prob, _, test_true_encoded_result,\
        test_approx_encoded_result = ml.build_pair_classifier_executer(1)
    ml.gather_gameplay_data(mnist_env, num_games)
    transition_seqs = ml.create_true_false_transition_dataset(n,
                                                          fraction_true=1/3)
    init = tf.global_variables_initializer()
    performupdate = U.function(ml.get_inputs(), losses, [update_steps[0]])
    performupdate_with_mse = U.function(ml.get_inputs(), losses, update_steps)
    saved_scopes = [ml.encoder_scope, ml.transition_scope, ml.equater_scope]

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
                if not stationary_distribution:
                    ml.gather_gameplay_data(mnist_env, num_games)
                    transition_seqs = ml.create_true_false_transition_dataset(
                        n, fraction_true=1/3)
                batch_losses = np.zeros(len(losses))
                l = 0
                for batch in tqdm(dataset.iterbatches(transition_seqs,
                                          batch_size=batch_size, shuffle=True)):
                    if l%50 == 0:
                        loss_results = performupdate_with_mse(*batch)
                    else:
                        loss_results = performupdate(*batch)
                    batch_losses += np.array(loss_results)
                    l +=1
                batch_losses = batch_losses/l
                for name, loss in zip(loss_names, batch_losses):
                    print(name + ": %.5f"%loss)
                if visualize:
                    t_inds = random.sample(range(n), 5)
                    t = [np.array([transition_seqs[i][j] for j in t_inds])
                                  for i in range(len(test_placeholders))]
                    l = np.array([transition_seqs[-1][j] for j in t_inds])
                    plt.pause(5)
                    plt.clf()
                    ml.visualize_transitions(
                        t, labels=l, get_prob=get_prob,
                        action_names=mnist_env.get_action_meanings())
                for sc in saved_scopes:
                    save_scope(
                        os.path.join(LOG_DIR, sc.name + "_weights.npz"), sc)
        if projecting:
            # IF TRAIN AND TEST FORMATS DIVERGE, CHANGE TO USE TRAIN DATASET
            visualize_embeddings(
                LOG_DIR,
                [test_true_encoded_result, test_approx_encoded_result],
                sess,
                transition_seqs[:-1],
                summary_writer=sw,
                labels=transition_seqs[-1],
                vis_mapping=[1, 1],
                data_placeholders=test_placeholders
            )
        sw.close()
        print("To visualize results, call:")
        print('tensorboard --logdir=%s' % LOG_DIR)

def debug_visualize_transition(reuse_weights=True):
    LOG_DIR = "./data/logdir"
    mnist_env = gym.make("mnist-v0")
    ml = ModelLearner(mnist_env.observation_space,
                                 mnist_env.action_space)
    ml.build_game_transition_classifier_cmp_only(pos_weight_multiplier=3)
    ml.gather_gameplay_data(mnist_env, 100)
    n = 1000
    transitions = ml.create_true_false_transition_dataset(
                n, fraction_true=1/3)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    save_dir = os.path.join(LOG_DIR, 'stationary_model')
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')

    with tf.Session() as sess:
        init.run()
        if reuse_weights:
            saver.restore(sess, checkpoint_path)
        while True:
            t_inds = random.sample(range(n), 5)
            t = [[transitions[i][j] for j in t_inds] for i in range(4)]
            ml.visualize_transition(t, sess=sess,
                                    action_names=mnist_env.get_action_meanings())
            plt.pause(1)
            plt.clf()

# debug_visualize_transition(False)
__main__()
# train_classic_mnist()
