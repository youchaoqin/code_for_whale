"""train a specific model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dataset_siamese
import math
import numpy as np
import cv2
import os
import feature_extractor
from build_loss import build_loss
import matplotlib.pylab as plt

slim = tf.contrib.slim

##### train configs #####
tf.app.flags.DEFINE_string('gpu', '0', 'CUDA_VISIBLE_DEVICES')
tf.app.flags.DEFINE_string(
    'cfg_file', None,
    'cfg file path, cfg file contains paremeters for training')
tf.app.flags.DEFINE_string(
    'output_dir', None,
    'output dir to save ckpts and summaries.')

FLAGS = tf.app.flags.FLAGS
#########################
#########################


def _configure_learning_rate(num_samples_per_epoch, global_step, train_cfg):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  lr_opt=train_cfg['lr_opt']
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.

  if lr_opt['lr_policy'] == 'exponential':
    decay_steps = int(num_samples_per_epoch * lr_opt.get('num_epochs_per_decay', 2.0) /
                      train_cfg['batch_size'])
    if train_cfg.get('sync_replicas', False):
      decay_steps /= train_cfg['replicas_to_aggregate']
    return tf.train.exponential_decay(lr_opt['learning_rate'],
                                      global_step,
                                      decay_steps,
                                      lr_opt['learning_rate_decay_factor'],
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif lr_opt['lr_policy'] == 'fixed':
    return tf.constant(lr_opt['learning_rate'], name='fixed_learning_rate')
  elif lr_opt['lr_policy'] == 'polynomial':
    return tf.train.polynomial_decay(lr_opt['learning_rate'],
                                     global_step,
                                     train_cfg['iters'],
                                     lr_opt['poly_end_lr'],
                                     power=lr_opt['poly_power'],
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  elif lr_opt['lr_policy'] == 'cosine':
      return tf.train.cosine_decay(learning_rate=lr_opt['learning_rate'],
                                   global_step=global_step,
                                   decay_steps=train_cfg['iters'],
                                   alpha=0.0,
                                   name='cosine_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     lr_opt['lr_policy'])


def _configure_optimizer(learning_rate, train_cfg):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  optimizer_opt = train_cfg['optimizer_opt']
  if optimizer_opt['optimizer'] == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=optimizer_opt['adadelta_rho'],
        epsilon=optimizer_opt.get('opt_epsilon', 1e-8))
  elif optimizer_opt['optimizer'] == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=optimizer_opt['adagrad_initial_accumulator_value'])
  elif optimizer_opt['optimizer'] == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=optimizer_opt['adam_beta1'],
        beta2=optimizer_opt['adam_beta2'],
        epsilon=optimizer_opt.get('opt_epsilon', 1e-8))
  elif optimizer_opt['optimizer'] == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=optimizer_opt['ftrl_learning_rate_power'],
        initial_accumulator_value=optimizer_opt['ftrl_initial_accumulator_value'],
        l1_regularization_strength=optimizer_opt['ftrl_l1'],
        l2_regularization_strength=optimizer_opt['ftrl_l2'])
  elif optimizer_opt['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=optimizer_opt['momentum'],
        name='Momentum')
  elif optimizer_opt['optimizer'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=optimizer_opt['rmsprop_decay'],
        momentum=optimizer_opt['rmsprop_momentum'],
        epsilon=optimizer_opt.get('opt_epsilon', 1e-8))
  elif optimizer_opt['optimizer'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % optimizer_opt['optimizer'])
  return optimizer


def _var_to_restore(exclude_scopes):
    if exclude_scopes is None:
        return slim.get_model_variables()

    model_variables = slim.get_model_variables()
    vars_to_restore = []
    ec_scopes = [s.strip() for s in exclude_scopes.split(',')]
    for mv in model_variables:
        flag = True
        for es in ec_scopes:
            if mv.op.name.startswith(es):
                flag = False
                break
        if flag:
            vars_to_restore.append(mv)
    return vars_to_restore


def _get_variables_to_train(train_cfg):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if train_cfg.get('trainable_scopes', None) is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in train_cfg['trainable_scopes'].split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

def _cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def build_siamese_distance(features, distance_type, scope='siamese_distance'):
    if len(features.shape) == 4:  # squeeze h, w
        features = tf.squeeze(features, axis=[1, 2])
    elif len(features.shape) == 2:
        pass
    else:
        raise Exception('shape of features is not right, '
                        'require 2 or 4,got: %s'%(features.shape))

    with tf.name_scope(scope):
        splited_features1, splited_features2 = tf.split(features, 2, axis=0)
        if distance_type == 'weighted_l1_distance':
            with tf.variable_scope(scope):
                alpha = tf.get_variable(
                    name='l1_alpha', shape=[1, features.shape[-1]], dtype=tf.float32,
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.2),
                    trainable=True,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.MODEL_VARIABLES])
                tf.summary.histogram('weighted_l1_distance/l1_alpha', alpha)
            distance = tf.subtract(splited_features1, splited_features2)
            distance = tf.math.abs(distance)
            distance = tf.reduce_sum(tf.multiply(distance, alpha), axis=-1, keepdims=True)
        else:
            raise Exception('distance not impelemented yet %s'%(distance_type))

    return distance


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
  if FLAGS.cfg_file is None:
    raise ValueError('You must supply the cfg file !')

  cfg = _cfg_from_file(FLAGS.cfg_file)
  train_cfg = cfg['train']

  # print all configs
  print('############################ cfg ############################')
  for k in cfg:
      print('%s: %s'%(k, cfg[k]))

  tf.logging.set_verbosity(tf.logging.INFO)
  #######################################################################
  ##############              sigle GPU version            ##############
  #######################################################################

  #### get dataset ####
  dataset_spec = _cfg_from_file(os.path.join(cfg['dataset_folder'], 'dataset_spec.yml'))
  siamese_dataset = dataset_siamese.get_siamese_dataset(
      dataset_folder=cfg['dataset_folder'],
      split=train_cfg['train_split'],
      cfg=train_cfg['dataset_opt'],
      is_training=True,
      shuffle=False,
      num_epoch=int(
          math.ceil(
              float(train_cfg['iters'] + 1) * train_cfg['batch_size'] /
              float(dataset_spec['num_examples'][train_cfg['train_split']]))),
  )

  #### build training dataset pipline #####
  im_batch, label_batch, _ = dataset_siamese.build_siamese_input_pipline(
      phase='train',
      dataset=siamese_dataset,
      min_resize_value=cfg.get('min_resize_value', None),
      max_resize_value=cfg.get('max_resize_value', None),
      # train cfgs:
      batch_size=train_cfg['batch_size'],
      shuffle=True,
      aug_opt=train_cfg.get('aug_opt', None),
      crop_size=cfg['corp_size'],
      dataset_spec=dataset_spec,
      drop_remainder=train_cfg.get('drop_remainder', True)  #since ites a siamese dataset
  )

  ##### get logits ####
  logits, endpoints = feature_extractor.extract_features(
      images=im_batch,
      num_classes=None,
      output_stride=cfg['output_stride'],
      global_pool=True,
      model_variant=cfg['model_variant'],
      weight_decay=train_cfg.get('weight_decy', 0.0),
      dropout_keep_prob=train_cfg.get('dropout_keep_prob', 1.0),
      regularize_depthwise=train_cfg.get('regularize_depthwise', False),
      reuse=tf.AUTO_REUSE,
      is_training=True,
      fine_tune_batch_norm=train_cfg.get('fine_turn_batch_norm', False),
      cfg=cfg)

  #### build distance and transform labels####
  distance = build_siamese_distance(
      features=logits,
      distance_type=cfg['siamese_distance'],
      scope='siamese_distance')

  ##### build loss ####
  total_loss = build_loss(  # default to use sigmoid ce for siamese network
      logits=distance,
      labels=label_batch,
      endpoints=endpoints,
      loss_opt=train_cfg['loss_opt'])

  #### build optiizer ####
  global_step = slim.create_global_step()
  learning_rate = _configure_learning_rate(
      num_samples_per_epoch=dataset_spec['num_examples'][train_cfg['train_split']],
      global_step=global_step,
      train_cfg=train_cfg)
  optimizer = _configure_optimizer(
      learning_rate=learning_rate,
      train_cfg=train_cfg,)

  #### build train tensor ####
  grads_and_vars = optimizer.compute_gradients(
      loss=total_loss,
      var_list=_get_variables_to_train(train_cfg=train_cfg),)
  grad_updates = optimizer.apply_gradients(
      grads_and_vars=grads_and_vars,
      global_step=global_step)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batch norm
  update_ops.append(grad_updates)
  update_op = tf.group(*update_ops)
  with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

  #### add summaries ####
  # Add summaries for model variables.
  for model_var in slim.get_model_variables():
      tf.summary.histogram(model_var.op.name, model_var)
  # Add summaries for losses.
  for loss in tf.get_collection(tf.GraphKeys.LOSSES):
      tf.summary.scalar('losses/%s' % loss.op.name, loss)
  if train_cfg['loss_opt'].get('use_reg_loss', False):
      tf.summary.scalar(
          'losses/reg_loss',
          tf.get_default_graph().get_tensor_by_name('make_total_loss/reg_loss:0'))
  if train_cfg['loss_opt'].get('use_aux_loss', False):
      tf.summary.scalar(
          'losses/aux_loss',
          tf.get_default_graph().get_tensor_by_name('make_total_loss/aux_loss/value:0'))
  tf.summary.scalar(
      'total_loss',
      tf.get_default_graph().get_tensor_by_name('make_total_loss/total_loss:0'))
  # merge all summaries
  merged_summaries = tf.summary.merge_all()
  summaries_writer = tf.summary.FileWriter(
      logdir=FLAGS.output_dir,
      graph=tf.get_default_graph())

  #### set up session config ####
  # savers:
  ckpt_saver = tf.train.Saver(max_to_keep=10)
  new_ckpt_path = os.path.join(FLAGS.output_dir, 'model.ckpt')
  save_ckpt_every = train_cfg.get('save_ckpt_every', 5000)
  # session config:
  sess_cfg = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
  sess_cfg.gpu_options.allow_growth = True

  #### train the model ####
  with tf.Session(config=sess_cfg) as sess:
      # init
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      # restore vars from pretrained ckpt:
      if train_cfg.get('pretrian_ckpt_file', None) is not None:
          pretrain_ckpt = train_cfg['pretrian_ckpt_file']
          tf.logging.info('restore ckpt from: %s', pretrain_ckpt)
          vars_to_restore = _var_to_restore(train_cfg.get('exclude_scopes', None))
          for v in vars_to_restore:
              print(v.op.name)
          restor_saver = tf.train.Saver(var_list=vars_to_restore)
          restor_saver.restore(sess, pretrain_ckpt)

      # train
      for i in range(train_cfg['iters']):
          if (i % save_ckpt_every == 0):
              all_summaries, loss_now = sess.run([merged_summaries, train_tensor])
              # write summaries
              summaries_writer.add_summary(all_summaries, i)
              # save ckpt
              ckpt_saver.save(sess, new_ckpt_path,global_step=i)
          else:
              loss_now = sess.run(train_tensor)
          if i % 20 == 0:
              tf.logging.info('global step: %d, loss= %f', i, loss_now)
      # Final run
      all_summaries, loss_now = sess.run([merged_summaries, train_tensor])
      # write summaries
      summaries_writer.add_summary(all_summaries, train_cfg['iters'])
      # save ckpt
      ckpt_saver.save(sess, new_ckpt_path, global_step=train_cfg['iters'])

  print("End of Train !!!")

  ## just for test the dataset
  # with tf.Session() as sess:
  #     sess.run(tf.global_variables_initializer())
  #     sess.run(tf.local_variables_initializer())
  #
  #     i = 0
  #     while True:
  #         try:
  #             ob_im, ob_label, ob_name = sess.run([im_batch, label_batch, im_name_batch])
  #             if i % 20 == 0:
  #                 # first pair
  #                 plt.subplot(2,2,1)
  #                 plt.imshow(np.squeeze(ob_im[0]).astype(np.uint8))
  #                 plt.title(ob_name[0].decode()+'_'+str(ob_label[0]))
  #                 plt.subplot(2, 2, 2)
  #                 plt.imshow(np.squeeze(ob_im[2]).astype(np.uint8))
  #                 plt.title(ob_name[2].decode() + '_' + str(ob_label[0]))
  #
  #                 # first pair
  #                 plt.subplot(2, 2, 3)
  #                 plt.imshow(np.squeeze(ob_im[1]).astype(np.uint8))
  #                 plt.title(ob_name[1].decode() + '_'+ str(ob_label[1]))
  #                 plt.subplot(2, 2, 4)
  #                 plt.imshow(np.squeeze(ob_im[3]).astype(np.uint8))
  #                 plt.title(ob_name[3].decode() + '_' + str(ob_label[1]))
  #
  #                 plt.show()
  #             i += 1
  #         except tf.errors.OutOfRangeError:
  #             print('End of Test !')
  #             break


if __name__ == '__main__':
  tf.app.run()
