import os
import argparse
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import configs

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
parser.add_argument('--logdir', default=None, help="Path to previously-created log"
                    "directory relative to ./, for resuming training")
parser.add_argument('--logname', default="default",
                    help="Experiment prefix for log directory, relative to ./data/env")
parser.add_argument('--configid', help="Config name string to use when "
                    "starting new training. Can be one of:\n"
                    "{}".format(list(configs.config_index.keys())))
parser.add_argument('--show-embeddings',
                    help="Project model progress using labelled embeddings",
                    action='store_true')
args = parser.parse_args()
scriptdir = os.path.dirname(os.path.realpath(__file__))
if args.logdir is not None:
    logdir = os.path.join(scriptdir, args.logdir)
    # logdir = args.logdir
    config = configs.load_config(logdir)
else:
    config = configs.get_config(args.configid)
    # logdir = os.path.join('data', config['env'], args.logname, 'train')
    logdir = os.path.join(scriptdir, 'data', config['env'], args.logname, 'train')
    logdir = increment_path(os.path.join(logdir, "run"))
    os.makedirs(logdir)
    configs.save_config(config, logdir)

import gym
import gym_mnist
import tensorflow as tf
from modellearner import ModelLearner

env = gym.make(config['env'])
logger.info("Logging results to {}".format(logdir))
sw = tf.summary.FileWriter(logdir)
ml = ModelLearner(env,
                  config,
                  summary_writer=sw)
saver = tf.train.Saver()
savepath = os.path.join(logdir, "model.ckpt")
restoring_saver = tf.train.Saver(var_list=[var for var in tf.global_variables()
                                           if var.name[:9] != "embedding"])
restore_path = tf.train.latest_checkpoint(logdir)
with tf.Session() as sess:
    if restore_path is not None:
        logger.info("Restoring variables from checkpoint: {}".format(restore_path))
        restoring_saver.restore(sess, restore_path)
    else:
        logger.info("Initializing brand new network parameters.")
        sess.run(tf.global_variables_initializer())
    global_step = sess.run(ml.global_step)
    logger.info("Gathering initial gameplay data!")
    if config['training_agent'] == "random_rollout":
        from agents import RandomRolloutAgent
        agent = RandomRolloutAgent(ml.envmodel)
        policy = agent.policy
    else:
        policy=None
    ml.gather_gameplay_data(config['n_initial_games'], policy=None)
    logger.info("Beginning training.")
    logger.info("To visualize, call:\ntensorboard --logdir={}".format(logdir))
    from utils import dataset
    while (not config['maxsteps']) or global_step < config['maxsteps']:
        transition_data = ml.create_transition_dataset(n=10000)
        if config['use_goal_boosting']:
            gb_data = ml.create_goals_dataset(n=1000)
            gb_data_batches = dataset.iterbatches(gb_data,
                                                  batch_size=config['batchsize'],
                                                  shuffle=True,
                                                  repeat=True)
        for batch in dataset.iterbatches(transition_data,
                                         batch_size=config['batchsize'],
                                         shuffle=True):
            if config['use_goal_boosting']:
                gb_batch = next(gb_data_batches)
            else:
                gb_batch = None
            ml.train_model(sess, *batch, gb_inputs=gb_batch)
        if args.show_embeddings:
            # Train a single step of ml.num_embed_vectors instances
            logger.info("Saving weights, creating embedding...")
            # create a dataset of max_horizon length transitions
            transition_data = ml.create_transition_dataset(n=ml.num_embed_vectors,
                                                           variable_steps=False)
            batch = next(dataset.iterbatches(transition_data,
                                             batch_size=ml.num_embed_vectors,
                                             shuffle=False))
            ml.train_model(sess, *batch, gb_inputs=gb_batch, show_embeddings=True)
            # By only saving during the embedding phase, our embedding isn't
            # overwritten by other saves
        else:
            logger.info("Saving weights...")
        saver.save(sess, savepath, global_step)
        global_step = sess.run(ml.global_step)
        ml.gather_gameplay_data(10, policy=policy)
logger.info("Training complete!")
