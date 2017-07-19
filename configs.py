# simple mnist single-goal config
import os
import dill as pickle
import json
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
    return pickle.load(open(picklepath, 'rb'))

config_index = {}
mnist_config_featureless = {
    'env': 'mnist-v0',
    'stepsize': 1e-4,
    'maxsteps': 10000000,
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': 128,
    'maxhorizon': 8,
    'x_to_f_ratio': 0,
    'x_to_g_ratio': 1,
    'transition_stacked_dim': 1,
    'minhorizon': 2,
    'batchsize': 16,
    'n_initial_games': 300,
    'use_goalstates': True,
}
config_index['mnist_simple'] = mnist_config_featureless

# simple multi-goal config (no features)
mnist_multigoal_config_featureless = {
    'env': 'mnist-multigoal-v0',
    'stepsize': 1e-4,
    'maxsteps': 10000000,
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': 128,
    'maxhorizon': 8,
    'x_to_f_ratio': 1,
    'x_to_g_ratio': 1,
    'batchsize': 16,
    'transition_stacked_dim': 1,
    'minhorizon': 2,
    'n_initial_games': 300,
    'use_goalstates': True,
}
config_index['mnist_multigoal'] = mnist_multigoal_config_featureless
# simple multi-goal config (w features)
mnist_multigoal_config_wfeature = {
    'env': 'mnist-multigoal-v0',
    'stepsize': 1e-4,
    'maxsteps': 10000000,
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1, 10], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'latent_size': 128,
    'maxhorizon': 8,
    'x_to_f_ratio': 1,
    'x_to_g_ratio': 1,
    'batchsize': 16,
    'transition_stacked_dim': 1,
    'minhorizon': 2,
    'n_initial_games': 300,
    'use_goalstates': True,
}
config_index['mnist_multigoal_wfeatures'] = mnist_multigoal_config_wfeature
# simplified mnist-9game config (fully observable)
mnist_9game_simple_wfeatures = {
    'env': 'mnist-9game-simple-v0',
    'stepsize': 1e-4,
    'maxsteps': 30000000,
    'feature_extractor': lambda state_info: state_info,
    'feature_shape': [3, 3, 3],
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: state_info[2][0],
    'has_labels': True,
    'latent_size': 128,
    'maxhorizon': 8,
    'x_to_f_ratio': 1,
    'x_to_g_ratio': 1,
    'batchsize': 16,
    'transition_stacked_dim': 1,
    'minhorizon': 2,
    'n_initial_games': 300,
    'use_goalstates': True,
}
config_index['9game_simple_wfeatures'] = mnist_9game_simple_wfeatures
