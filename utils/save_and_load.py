import tensorflow as tf
import numpy as np
import sys, os

## run from within desired session!

def save_scope(outfile, scope):
    all_vars = get_scope_vars(scope)
    names = []
    weights = []
    weight_dict = {}
    for var in all_vars:
        names.append(var.name)
        weights.append(var.eval())
        weight_dict[var.name] = var.eval()
        # print(var.name, var.eval())
    np.savez(outfile, **weight_dict)

def get_scope_vars(scope):
    if isinstance(scope, str):
        scopename = scope
    else:
        scopename = scope.name
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=scopename)

def load_scope(infile, scope):
    ## Get pretrained scaled weights into graph.
    all_vars = get_scope_vars(scope)
    weight_dict = {}
    for var in all_vars:
        weight_dict[var.name] = var
    loaded_weights = np.load(infile)
    keys = sorted(loaded_weights.keys())
    for k in keys:
        try:
            weight_dict[str(k)].assign(loaded_weights[k]).eval()
        except Exception as e:
            print("Couldn't assign weights", e, k)
