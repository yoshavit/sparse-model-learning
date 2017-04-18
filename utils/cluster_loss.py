import tensorflow as tf

def cluster_loss(input1, input2, is_same, pos_weight, name=None):
    """Computes a special loss function that penalizes input1 and input2 if
    for being far if is_same=True, and for being close if is_same=False

    If positive, loss = (input1 - input2)^2
    If negative, loss = exp(-|input1 - input2|)

    Args:
        input1 - data vector of size ?xN
        input2 - data vector of size ?xN
        is_same - a boolean vector of size ?x1, True iff input1 and input2
            should be in the same cluster
        pos_weight - how much more we should weight the positive loss over
            the negative example loss, as a scalar

    Returns:
        loss - a loss tensor of size ?x1

    """
    with tf.variable_scope(name if name else "cluster_loss") as scope:
        d = tf.reduce_mean(tf.squared_difference(input1, input2), axis=1)
        positive_loss = pos_weight*d
        negative_loss = tf.exp(-d)
        loss = tf.where(is_same, positive_loss, negative_loss, name="loss")
        return loss

