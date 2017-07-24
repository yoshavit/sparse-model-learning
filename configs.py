# simple mnist single-goal config
import os
import dill as pickle
import json
config_index = {}
default_params = {
# Parameters meant to make older paramfiles forwards-compatible
# Must all be primitives, because we will make a shallow copy
    'stepsize': 1e-4,
    'maxsteps': 1e7,
    'has_labels': False,
    'latent_size': 64,
    'x_to_f_ratio': 1,
    'x_to_g_ratio': 1,
    'maxhorizon':8,
    'sigmoid_latents': False,
    'use_goalstates': True,
    'minhorizon': 2,
    'transition_stacked_dim': 1,
    'training_agent': 'random',
    'n_initial_games': 200,
    'batchsize': 16
}
def isalambda(v):
    LAMBDA = lambda:0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
def save_config(config, savedir):
    picklepath = os.path.join(savedir, "config.p")
    jsonpath = os.path.join(savedir, "config.json")
    pickle.dump(config, open(picklepath, 'wb'))
    lambdaless_config = {}
    for k, v in config.items():
        if not isalambda(v):
            lambdaless_config[k] = v
    json.dump(lambdaless_config, open(jsonpath, 'w'))
def load_config(loaddir):
    picklepath = os.path.join(loaddir, "config.p")
    config =  pickle.load(open(picklepath, 'rb'))
    for k in default_params.keys(): # make backwards compatible
        if k not in config.keys():
            config[k] = default_params[k]
    return pickle.load(open(picklepath, 'rb'))
def get_config(configid):
    " Create a new config, indexed by the keys in config_index"
    config = default_params.copy()
    try:
        custom_config = config_index[configid]
    except KeyError:
        raise KeyError("configid must be one of {}".format(config_index.keys()))
    for k,v in custom_config.items():
        config[k] = v
    return config

mnist_config_featureless = {
    'env': 'mnist-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'x_to_f_ratio': 0,
}
config_index['mnist_simple'] = mnist_config_featureless

# simple multi-goal config (no features)
mnist_multigoal_config_featureless = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'x_to_f_ratio': 0,
}
config_index['mnist_multigoal'] = mnist_multigoal_config_featureless
# simple mnist config with linear action dynamics
mnist_linear_config_featureless_shorthorizon = {
    'env': 'mnist-linear-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2],
    'feature_type': "softmax",
    'x_to_f_ratio': 0,
    'has_labels': True,
    'label_extractor': lambda state_info: [state_info],
    'sigmoid_latents': True,
    'maxhorizon': 3,
    'minhorizon': 1,
}
config_index['mnist_linear_nfeat_fewstep'] = mnist_linear_config_featureless_shorthorizon
# simple multi-goal config (no features, yes sigmoided latents and an agent that
# uses learning to explore)
mnist_multigoal_config_nfeature_wsig_wagent = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info==0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'x_to_f_ratio': 0,
    'sigmoided_latents': True,
    'has_labels': True,
    'training_agent': 'random_rollout'
}
config_index['mnist_multigoal_nfeat_wsig_wagent'] = mnist_multigoal_config_nfeature_wsig_wagent
# simple multi-goal config (w features and sigmoided latents)
mnist_multigoal_config_wfeature_wsig = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1, 10], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'sigmoided_latents': True,
    'has_labels': True,
}
config_index['mnist_multigoal_wfeat_wsig'] = mnist_multigoal_config_wfeature_wsig
# simple multi-goal config (w features and sigmoided latents and an agent that
# uses learning to explore)
mnist_multigoal_config_wfeature_wsig_wagent = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1, 10], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'sigmoided_latents': True,
    'training_agent': 'random_rollout',
}
config_index['mnist_multigoal_wfeat_wsig_wagent'] = mnist_multigoal_config_wfeature_wsig_wagent
# simplified mnist-9game config (fully observable)
mnist_9game_simple_wfeatures = {
    'env': 'mnist-9game-simple-v0',
    'feature_extractor': lambda state_info: state_info,
    'feature_shape': [3, 3, 3],
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: state_info[2][0],
    'has_labels': True,
}
config_index['9game_simple_wfeatures'] = mnist_9game_simple_wfeatures
