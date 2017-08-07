# simple mnist single-goal config
import os
import dill as pickle
import json
config_index = {}
default_params = {
# Parameters meant to make older paramfiles forwards-compatible
# Must all be primitives, because we will make a shallow copy
    'stepsize': 1e-4,
    'maxsteps': 5e6,
    'has_labels': False,
    'latent_size': 128,
    'f_scalar': 1,
    'g_scalar': 1,
    'x_scalar': 1,
    'maxhorizon':8,
    'sigmoid_latents': False,
    'use_goalstates': True,
    'minhorizon': 1,
    'transition_stacked_dim': 1,
    'training_agent': 'random',
    'n_initial_games': 200,
    'batchsize': 16,
    'use_goal_boosting': False,
    'x_to_gb_ratio': 1,
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



config = {
    'env': 'mnist-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'f_scalar': 0,
}
config_index['mnist_simple'] = config
# mnist with 2 layers
config = {
    'env': 'mnist-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
            # label_extractor - (optional) function from info['state'/'next_state']
                # to label. If provided, output includes a fourth column, "labels"
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'maxhorizon': 6,
    'transition_stacked_dim': 2,
    'f_scalar': 0,
}
config_index['mnist_simple_3lt'] = config.copy()
config.update({'env': 'mnist-linear-v0'})
config_index['mnist_linear_3lt'] = config.copy()
# simple multi-goal config (no features)
config = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'f_scalar': 0,
}
config_index['mnist_multigoal'] = config

# ------------------ LINEAR HORIZON-VARYING -----------------------
basename = "mnist_linear_"
# simple mnist config with linear action dynamics
config = {
    'env': 'mnist-linear-v0',
    'feature_extractor': lambda state_info: [state_info == 0],
    'feature_shape': [1, 2],
    'feature_type': "softmax",
    'transition_stacked_dim': 1,
    'f_scalar': 0,
    'has_labels': True,
    'label_extractor': lambda state_info: [state_info],
    'x_to_gb_ratio': 0.5,
}
for i in range(1, 7):
    for use_gb in [True, False]:
        config.update({'maxhorizon': i, 'use_goal_boosting': use_gb})
        name = basename + '%dstep'%i + use_gb*'_wgb'
        config_index[name] = config.copy()
config.update({'maxhorizon': 2, 'use_goal_boosting': True})
for i in range(0, 7):
    config.update({'x_scalar': 10.0**(-i)})
    name = basename + 'xe-%d'%i
    config_index[name] = config.copy()
config.update({'use_goal_boosting': False, 'env': 'mnist-v0'})
# ----------- MNIST REGULAR HORIZON VARYING --------------
basename = 'mnist_simple_'
for i in range(1, 9):
    config.update({'maxhorizon': i})
    name = basename + '%dstep'%i
    config_index[name] = config.copy()
# ----------- MNIST WFEATURES FEATURE-DENSITY-VARYING ------------
config = {
    'env': 'mnist-v0',
    'maxhorizon': 3,
    'has_labels': True,
    'f_scalar': 1,
    'label_extractor': lambda state_info: [state_info],
}
basename = 'mnist_wfeat_complex_'
for i in [2, 3, 4, 5, 10]:
    config.update({
        'feature_extractor': lambda state_info: [state_info%i],
        'feature_type': 'softmax',
        'feature_shape': [1, i]})
    name = basename + 'mod%d'%i
    config_index[name] = config.copy()

# -----------------------------------------------------------------
# simple multi-goal config (no features, yes sigmoided latents and an agent that
# uses learning to explore)
config = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info==0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'f_scalar': 0,
    'sigmoid_latents': True,
    'has_labels': True,
    'x_to_gb_ratio': 0.5,
    'use_goal_boosting': True
}
config_index['mnist_multigoal_nfeat_wsig_wgb'] = config
# simple multi-goal config (w features and sigmoided latents)
config = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1, 10], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'sigmoid_latents': True,
    'has_labels': True,
}
config_index['mnist_multigoal_wfeat_wsig'] = config

# simple linear config without latent consistency
config = {
    'env': 'mnist-linear-v0',
    'feature_extractor': lambda state_info: [state_info==0],
    'feature_shape': [1, 2], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'maxhorizon': 3,
    'f_scalar' : 0,
    'x_scalar' : 0,
}
config_index['mnist_linear_3step_noxloss'] = config
# simple multi-goal config (w features and sigmoided latents and an agent that
# uses learning to explore)
config = {
    'env': 'mnist-multigoal-v0',
    'feature_extractor': lambda state_info: [state_info],
    'feature_shape': [1, 10], # one feature, with two possible classes
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: [state_info],
    'has_labels': True,
    'training_agent': 'random_rollout',
}
config_index['mnist_multigoal_wfeat_wagent'] = config
# simplified mnist-9game config (fully observable)
config = {
    'env': 'mnist-9game-simple-v0',
    'feature_extractor': lambda state_info: state_info,
    'feature_shape': [3, 3, 3],
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: state_info[2][0],
    'has_labels': True,
}
config_index['9game_wfeat'] = config
# 3-column mnist flipgame (with only 0s and 1s)
config = {
    'env': 'flipgame-v0',
    'feature_extractor': lambda state_info: state_info,
    'feature_shape': [3, 3, 2],
    'feature_type': 'softmax',
    'label_extractor': lambda state_info: state_info[2][0],
    'has_labels': True,
}
config_index['flipgame_wfeat'] = config
