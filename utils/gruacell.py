import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops import array_ops, nn_ops, math_ops, init_ops, variable_scope as vs
from tensorflow.python.util import nest
# Taken from parameters at https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py 
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class GRUACell(rnn.GRUCell):
    def __init__(self, num_units, num_factors, input_size=None, activation=tanh,
                 reuse=None):
        super(GRUACell, self).__init__(num_units, input_size, activation,
                                       reuse)
        self.num_factors = num_factors

    def call(self, inputs, state):
        """Gated recurrent unit with Actions (GRUA) with nunits cells."""
        # =========================== MODIFIED ===========================
        # Using the Oh et al. 2015 model to combine actions and states
        with vs.variable("factors"):
            action_factors = _linear(inputs, self.num_factors, False,
                                     self._bias_initializer,
                                     self._kernel_initializer)
            state_factors = _linear(state, self.num_factors, False,
                                    self._bias_initializer,
                                    self._kernel_initializer)
            factors = action_factors*state_factors
        # ===============================================================
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [factors, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = sigmoid(_linear([factors, state], 2 * self._num_units, True,
                                    bias_ones, self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(_linear([factors, r * state], self._num_units,
                                         True, self._bias_initializer, self._kernel_initializer))
        new_h = u * state + (1 - u) * c
        return new_h, new_h

# Taken from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
def _linear(args, output_size, bias, bias_initializer=None,
                        kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias; None by default.
        kernel_initializer: starting value to initialize the weight; None by default.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype,
                initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                    _BIAS_VARIABLE_NAME, [output_size],
                    dtype=dtype,
                    initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)
