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
parser.add_argument('--no-train', help="Do not update the parameters",
                    action="store_true")
parser.add_argument('--ignore-latent-consistency',
                    help="Penalize latent encodings' consistency with each other via MSE",
                    action="store_true")
parser.add_argument('-batchsize', help="Batch size for learning", type=int,
                    default=16)
parser.add_argument('--maxhorizon', help="Max number of steps to simulate"
                    "forward (1 means 1 transition)", type=int, default=5)
parser.add_argument('--maxsteps', type=int, default=10000000, help="Number of"
                    "steps to train (if 0, trains until manually halted)")
parser.add_argument('-stepsize', type=float, default=1e-4,
                    help="train step size")
parser.add_argument('--latentsize', type=int, default=32,
                    help="Number of latent dimensions used to encode the state")

args = parser.parse_args()
env = gym.make(args.env)
max_horizon = args.maxhorizon
# ------ MNIST features ------------
# Feature will be true MNIST digit
# Feature is extracted from info
# feature_extractor = lambda info: [info%5]
# feature_size = 1
# Label is separate from info
# label_extractor = lambda info: [info]
# ------- 9-game features---------------
feature_extractor = lambda info: info.flatten()
feature_size = 9
label_extractor = None
# ----------------------------------
has_labels = bool(label_extractor)
logdir = os.path.join('data', args.env, args.logdir, 'train')
if args.run_id:
    logdir = os.path.join(logdir, args.run_id)
else: # increment until we get a new id
    logdir = increment_path(os.path.join(logdir, "run"))
logger.info("Logging results to {}".format(logdir))
sw = tf.summary.FileWriter(logdir)
ml = ModelLearner(env, feature_size, args.stepsize,
                  latent_size=args.latentsize,
                  max_horizon=max_horizon,
                  summary_writer=sw,
                  has_labels=has_labels,
                  force_latent_consistency=(not
                                            args.ignore_latent_consistency),
                  feature_extractor=feature_extractor)
saver = tf.train.Saver()
savepath = os.path.join(logdir, "model.ckpt")
logger.info("Gathering initial gameplay data!")
ml.gather_gameplay_data(200)
restoring_saver = tf.train.Saver(var_list=[var for var in tf.global_variables()
                                           if var.name[:9] != "embedding"])
local_init_op = tf.global_variables_initializer()
sv = tf.train.Supervisor(
    saver=restoring_saver,
    logdir=logdir,
    local_init_op=local_init_op,
    summary_op=None,
    summary_writer=sw,
    global_step=ml.global_step,
    save_model_secs=0,
    save_summaries_secs=20)
with sv.managed_session() as sess:
    global_step = sess.run(ml.global_step)
    logger.info("Beginning training.")
    logger.info("To visualize, call:\ntensorboard --logdir={}".format(logdir))
    while not sv.should_stop() and ((not args.maxsteps) or global_step <
                                    args.maxsteps):
        transition_data = ml.create_transition_dataset(max_horizon,
                                                       n=20000,
                                                       label_extractor=label_extractor)
        for batch in dataset.iterbatches(transition_data,
                                         batch_size=args.batchsize,
                                         shuffle=True):
            ml.train_model(sess, *batch)
        if args.show_embeddings:
            # Train a single step of ml.num_embed_vectors instances
            logger.info("Creating embedding...")
            # create a dataset of max_horizon length transitions
            transition_data = ml.create_transition_dataset(max_horizon,
                                                           n=1000,
                                                           label_extractor=label_extractor,
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
