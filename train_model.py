import os
import argparse
import gym
import gym_mnist
import tensorflow as tf
import logging
from utils import dataset
from modellearner import ModelLearner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
parser.add_argument('env', default="mnist-v0", help="Name of environment to"
                    "be trained on. Examples: 'mnist-v0', 'blockwalker-v0',"
                    "and 'blockwalker-multicolored-v0'")
parser.add_argument('--logdir', help="Path to log directory, relative to ./data/[env]")
parser.add_argument('--run-id', help="Log suffix pointing to experiment to"
                    "resume, e.g. if was run--05 arg should be run--05",
                    type=str)
parser.add_argument('--show-embeddings', help="Project model progress using labelled embeddings",
                    action='store_true')
# parser.add_argument('--ignore-latent-consistency',
                    # help="Penalize latent encodings' consistency with each other via MSE",
                    # action="store_true")
parser.add_argument('-batchsize', help="Batch size for learning", type=int,
                    default=16)
parser.add_argument('--maxhorizon', help="Max number of steps to simulate"
                    "forward (1 means 1 transition)", type=int, default=5)
parser.add_argument('--maxsteps', type=int, default=10000000, help="Number of"
                    "steps to train (if 0, trains until manually halted)")
parser.add_argument('-stepsize', type=float, default=1e-4,
                    help="train step size")
parser.add_argument('--latentsize', type=int, default=64,
                    help="Number of latent dimensions used to encode the state")

args = parser.parse_args()
# ----------------------------------------------
"""
Things needed in config file:
"""
mnist_config = {
    'env': 'mnist-v0',
    'logdir': args.logdir,
    'stepsize': args.stepsize,
    'maxsteps': args.maxsteps,
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1],
    'feature_regression': True,
    'feature_softmax': False,
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': args.latentsize,
    'maxhorizon': args.maxhorizon,
    'force_latent_consistency': True,
    'transition_stacked_dim': 1,
    'minhorizon': 1,
    'n_initial_games': 300,
    'use_goalstates': True,
}
# simple multi-goal config
mnist_multigoal_config = {
    'env': 'mnist-multigoal-v0',
    'stepsize': args.stepsize,
    'maxsteps': args.maxsteps,
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1],
    'feature_regression': True,
    'feature_softmax': False,
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': args.latentsize,
    'maxhorizon': args.maxhorizon,
    'force_latent_consistency': True,
    'transition_stacked_dim': 1,
    'minhorizon': 1,
    'n_initial_games': 500,
    'use_goalstates': True,
}
config = mnist_config

env = gym.make(config['env'])


# ------ MNIST features ------------
# Feature will be true MNIST digit
# Feature is extracted from info['state']
feature_extractor = lambda state_info: [state_info%5]
feature_shape = [1]
# Label is separate from info['state']
label_extractor = lambda state_info: [state_info]
feature_regression = True
feature_softmax = False
# ------- 9-game features---------------
# feature_extractor = lambda info: info
# feature_shape = [3, 3, 9]
# label_extractor = lambda info: info[2][0]
# feature_regression = False
# feature_softmax = True
# ----------------------------------
has_labels = bool(label_extractor)
logdir = os.path.join('data', config['env'], config['logdir'], 'train')
if args.run_id:
    logdir = os.path.join(logdir, args.run_id)
else: # increment until we get a new id
    logdir = increment_path(os.path.join(logdir, "run"))
logger.info("Logging results to {}".format(logdir))
sw = tf.summary.FileWriter(logdir)
ml = ModelLearner(env,
                  config,
                  summary_writer=sw)
saver = tf.train.Saver()
savepath = os.path.join(logdir, "model.ckpt")
logger.info("Gathering initial gameplay data!")
ml.gather_gameplay_data(config['n_initial_games'])
restoring_saver = tf.train.Saver(var_list=[var for var in tf.global_variables()
                                           if var.name[:9] != "embedding"])
local_init_op = tf.global_variables_initializer()
restore_path = tf.train.latest_checkpoint(logdir)
with tf.Session() as sess:
    if restore_path is not None:
        logger.info("Restoring variables from checkpoint: {}".format(restore_path))
        restoring_saver.restore(sess, restore_path)
    else:
        logger.info("Initializing brand new network parameters.")
        sess.run(local_init_op)
    global_step = sess.run(ml.global_step)
    logger.info("Beginning training.")
    logger.info("To visualize, call:\ntensorboard --logdir={}".format(logdir))
    while (not args.maxsteps) or global_step < config['maxsteps']:
        transition_data = ml.create_transition_dataset(n=20000)
        for batch in dataset.iterbatches(transition_data,
                                         batch_size=args.batchsize,
                                         shuffle=True):
            ml.train_model(sess, *batch)
        if args.show_embeddings:
            # Train a single step of ml.num_embed_vectors instances
            logger.info("Creating embedding...")
            # create a dataset of max_horizon length transitions
            transition_data = ml.create_transition_dataset(n=ml.num_embed_vectors,
                                                           variable_steps=False)
            batch = next(dataset.iterbatches(transition_data,
                                             batch_size=ml.num_embed_vectors,
                                             shuffle=False))
            ml.train_model(sess, *batch, show_embeddings=True)
            # By only saving during the embedding phase, our embedding isn't
            # overwritten by other saves
            saver.save(sess, savepath, global_step)
        global_step = sess.run(ml.global_step)
        ml.gather_gameplay_data(10)
logger.info("Training complete!")
