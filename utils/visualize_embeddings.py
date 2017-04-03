import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import scipy.misc
from utils import dataset

def visualize_embeddings(logdir, target_tensors, sess, data,
                               labels=None,
                               vis_mapping=None,
                               data_placeholders=None,
                               summary_writer=None):
    """Creates all relevant files to visualize MNIST digit embeddings using
    tensorboard --logdir=LOG_DIR

    Args:
        target_tensors: ?xD tensors containing the desired embedding
        vis_mapping - list of integers, for each in target_tensors the
            index of the relevant data vector
        data - tensors containing data to be fed in 
    """
    print ("Creating embedding")
    for i in range(len(data)): data[i] = np.array(data[i])
    if labels is not None:
        labels = np.array(labels)
    if not summary_writer:
        summary_writer = tf.summary.FileWriter(logdir)
    if not vis_mapping:
        vis_mapping = [0 for i in range(len(target_tensors))] # use first entry
    config = projector.ProjectorConfig()
    inputs = data
    placeholders = data_placeholders
    embedding_values = do_elementwise_eval(target_tensors, placeholders,
                                           inputs)
    embed_vars = []
    for i in range(len(embedding_values)):
        embed_var = tf.Variable(np.array(embedding_values[i]),
                                name="layer_%d"%i)
        embed_vars.append(embed_var)
        embed_var.initializer.run()
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_var.name
        embedding.sprite.image_path = os.path.join(logdir,
                                                   'embed_sprite%d.png'%i)
        image_data = data[vis_mapping[i]]
        thumbnail_size = image_data.shape[1]
        embedding.sprite.single_image_dim.extend([thumbnail_size,
                                                  thumbnail_size])
        sprite = images_to_sprite(image_data)
        scipy.misc.imsave(embedding.sprite.image_path, sprite)
    saver = tf.train.Saver(embed_vars)
    saver.save(sess, os.path.join(logdir, 'embed_model.ckpt'))
    if labels is not None:
        embedding.metadata_path = os.path.join(logdir, 'embed_labels.tsv')
        metadata_file = open(embedding.metadata_path, 'w')
        metadata_file.write('Name\tClass\n')
        for ll in range(len(labels)):
            metadata_file.write('%06d\t%d\n' % (ll, labels[ll]))
        metadata_file.close()

    projector.visualize_embeddings(summary_writer, config)
    print("Embedding created.")

def do_elementwise_eval(output_tensors, placeholders, inputs):
    """Evaluates the desired tensors using the data/labels, by breaking the
    computation up into batches.
    Args:
        output_tensors - tensors to evaluate and provide in the output
        placeholders - list of input placeholders
        inputs - list of data to be inputted, same order as placeholders
    """
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    if not isinstance(inputs, list):
        inputs = [inputs]
    n = inputs[0].shape[0]
    batch_size = 32
    all_outputs = [[] for tensor in output_tensors]
    for batch_inputs in dataset.iterbatches(inputs, batch_size=batch_size,
                                            shuffle=False):
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[placeholders[i]] = batch_inputs[i]
        for i in range(len(output_tensors)):
            tensor = output_tensors[i]
            output_value = tensor.eval(feed_dict=feed_dict)
            all_outputs[i].append(output_value)
    for k in range(len(all_outputs)):
        all_outputs[k] = np.concatenate(all_outputs[k])
    return all_outputs

# Taken from https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


