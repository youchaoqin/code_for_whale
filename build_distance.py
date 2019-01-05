import tensorflow as tf
import tensorflow.contrib.slim as slim


def compute_distance(features_a, features_b, d_cfg, is_training=False):
    """
    compute_distance for features_a, features_b,
    they have same shape: [num_image, feature_long]
    """
    if d_cfg['distance_type'] == 'weighted_l1_distance':
        # accroding to "Siamese neural networks for one-shot image recognition"
        # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
        with tf.name_scope(d_cfg['distance_type']):
            distances = tf.subtract(features_a, features_b)
            distances = tf.math.abs(distances)
            # get alpha and use it to weight distances
            with tf.variable_scope(d_cfg['distance_type']):
                alpha = tf.get_variable(
                    name='l1_alpha', shape=[1, distances.shape[-1]], dtype=tf.float32,
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.2),
                    trainable=True,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.MODEL_VARIABLES])
                tf.summary.histogram('weighted_l1_distance/l1_alpha', alpha)
            distances = tf.reduce_sum(
                tf.multiply(distances, alpha), axis=-1, keepdims=True)
    elif d_cfg['distance_type'] == 'learnable_fc_x3':
        # accroding to learning to compare: relation networks for few-shot learning
        # https://arxiv.org/abs/1711.06025
        with tf.variable_scope(d_cfg['distance_type']):
            concated_features = tf.concat([features_a, features_b], axis=-1)
            net = slim.dropout(concated_features, keep_prob=0.5, is_training=is_training)
            net = slim.fully_connected(
                inputs=net,
                num_outputs=1024,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(0.00001),
                biases_initializer=tf.zeros_initializer(),
                scope='fc1')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            net = slim.fully_connected(
                inputs=net,
                num_outputs=1024,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(0.00001),
                biases_initializer=tf.zeros_initializer(),
                scope='fc2')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            distances = slim.fully_connected(
                inputs=net,
                num_outputs=1,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(0.00001),
                biases_initializer=tf.zeros_initializer(),
                scope='fc3')
    else:
        raise Exception('un-known distance type: %s' % (d_cfg['distance_type']))
    return distances


def build_distance_for_pairs_batch(
        features, d_cfg, is_training=False, scope='build_distance'):
    """
    compute distance for batch of image pairs, assume the first image of a pair is in the
    first half of the batch and the second image of this pair is in the second half of the
    batch
    """
    if len(features.shape) == 4:  # squeeze h, w
        features = tf.squeeze(features, axis=[1, 2])
    elif len(features.shape) == 2:
        pass
    else:
        raise Exception('shape of features is not right, '
                        'require 2 or 4,got: %s'%(features.shape))

    with tf.name_scope(scope):
        splited_features1, splited_features2 = tf.split(features, 2, axis=0)
        distances = compute_distance(
            features_a=splited_features1, features_b=splited_features2,
            d_cfg=d_cfg, is_training=is_training,)


    return distances