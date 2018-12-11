import tensorflow as tf
import numpy as np
import os

def _focal_loss_alpha_from_file(fl_alpha_file):
    """
    :param fl_alpha_file: a .npy file
    :return: a tf.constant of alpha
    """
    fl_alpha = np.load(fl_alpha_file)
    fl_alpha = np.expand_dims(fl_alpha, axis=0)
    focal_loss_alpha = tf.constant(fl_alpha, dtype=tf.float32, shape=fl_alpha.shape)
    return focal_loss_alpha

def class_weighted_focal_loss(onehot_labels, logits, gamma, alpha):
    with tf.name_scope('class_weighted_focal_loss'):
        # per example alpha
        alpha_reshaped = tf.reshape(alpha, shape=[1, onehot_labels.shape[-1]])
        per_example_alpha = tf.multiply(onehot_labels, alpha_reshaped)
        per_example_alpha = tf.reduce_sum(per_example_alpha, axis=-1, keepdims=True)

        # per example hard-example cross-entropy
        per_example_prob = tf.nn.softmax(logits, axis=-1)
        per_example_prob = tf.multiply(per_example_prob, onehot_labels)
        per_example_prob = tf.reduce_sum(per_example_prob, axis=-1, keepdims=True)
        per_example_weight = tf.pow((1.0-per_example_prob), gamma)  # hard example mining

        # focal loss
        focal_loss = -per_example_alpha * per_example_weight * tf.log(per_example_prob)
        focal_loss = tf.reduce_mean(focal_loss)
        tf.losses.add_loss(focal_loss)  # add to the tf.GraphKey.LOSSES
        return focal_loss


def build_loss(logits, labels, endpoints, loss_opt):
    with tf.name_scope('make_total_loss'):
        if len(logits.shape) == 4:
            logits_in = tf.squeeze(logits,axis=[-3, -2])
        elif len(logits.shape) == 2:
            logits_in = tf.identity(logits)
        else:
            raise Exception('logits shape not right: %s'%(logits.shape))

        # build main loss
        main_loss_type = loss_opt['main_loss_type']
        if main_loss_type == 'softmax_cross_entropy':
            tf.logging.info('### use softmax_cross_entropy ###')
            total_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits_in,
                weights=loss_opt.get('main_loss_weight', 1.0),
                scope='main_loss')
        elif main_loss_type == 'class_weighted_focal_loss':
            tf.logging.info('### use class_weighted_focal_loss ###')
            focal_loss_alpha_file = loss_opt.get('focal_loss_alpha_file', None)
            if focal_loss_alpha_file is not None:
                focal_loss_alpha = _focal_loss_alpha_from_file(focal_loss_alpha_file)
            else:
                fl_alpha = loss_opt.get('focal_loss_alpha', None)
                if fl_alpha is None:  # no class-balance
                    tf.logging.info('### No class-balance, use all 1.0 ###')
                    fl_alpha_long = logits.shape[-1]
                    fl_alpha_long = fl_alpha_long.value
                    fl_alpha = [1.0 for _ in range(fl_alpha_long)]
                focal_loss_alpha = tf.constant(
                    np.expand_dims(np.array(fl_alpha), axis=0),
                    dtype=tf.float32, shape=[1, len(fl_alpha)])
            total_loss = class_weighted_focal_loss(
                onehot_labels=labels, logits=logits_in,
                gamma=loss_opt.get('focal_loss_gamma', 2.0),
                alpha=focal_loss_alpha)
        else:
            raise Exception('Un-known main loss type')

        # add aux loss
        if loss_opt.get('use_aux_loss', False):
            tf.logging.info('### also use aux_loss ###')
            try:
                aux_logits = tf.squeeze(endpoints['aux_logits'])
            except:
                raise Exception("Don't have aux_logits in endpoints!")
            aux_loss_type = loss_opt.get('aux_loss_type', 'softmax_cross_entropy')
            if aux_loss_type == 'softmax_cross_entropy':
                aux_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=aux_logits,
                weights=loss_opt.get('aux_loss_weight', 1.0),
                scope='aux_loss')
            else:
                raise Exception('Un-known aux loss type')
            total_loss = tf.add(total_loss, aux_loss)

        # add regularization loss
        if loss_opt.get('use_reg_loss', False):
            tf.logging.info('### also use reg_loss ###')
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(reg_loss, 'reg_loss')
            total_loss = tf.add(total_loss, reg_loss)

        total_loss = tf.debugging.check_numerics(total_loss,
                                                 'total_loss is NaN or Inf!')
        total_loss = tf.identity(total_loss, 'total_loss')
    return total_loss