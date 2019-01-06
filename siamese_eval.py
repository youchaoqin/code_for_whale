"""eval a specific model using a given dataset.
either eval once or eval repeatedly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.training.python.training.evaluation import checkpoints_iterator

import dataset_siamese
import math
import numpy as np
import cv2
import os
import feature_extractor
from build_loss import build_loss
from build_distance import build_distance_for_pairs_batch


slim = tf.contrib.slim

##### train configs #####
tf.app.flags.DEFINE_string('gpu', '0', 'CUDA_VISIBLE_DEVICES')
tf.app.flags.DEFINE_string(
    'cfg_file', None,
    'cfg file path, cfg file contains paremeters for training')
tf.app.flags.DEFINE_string(
    'ckpt_dir', None,
    'dir to load ckpts and save summaries in ckpt_dir/eval.')
tf.app.flags.DEFINE_bool(
    'eval_once', True,
    'eval the newest ckpt once, if False, eval when a new ckpt available')
tf.app.flags.DEFINE_integer(
    'min_interval_secs', 60*30,
    'the minimum seconds interval to yield another ckpt')
tf.app.flags.DEFINE_integer(
    'ckpt_timeout_sec', 5000,
    'seconds to wait a new ckpt')
tf.app.flags.DEFINE_bool(
    'restore_global_step', False,
    'wether restore global step, for logging a series evaluations'
)
FLAGS = tf.app.flags.FLAGS
#########################
#########################
def _time_out_fn(sec):
    tf.logging.info('no ckpt found after waiting for %d seconds', sec)
    return True


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


def _cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def make_metric_dict(prob, label, metric_opt):
    """
    Args:
        :param prob: [batch, num_cls]
        :param label: [batch, num_cls]
        :param metric_opt: metric options
    :return: metirc_dict, dict contain
    """
    metric_dict = {}
    with tf.name_scope('make_metrics_dict'):
        eval_metrics = metric_opt.get('eval_metrics', ['accuracy'])
        for em in eval_metrics:
            if em.strip() == 'accuracy':
                # accuracy:
                _, acc_up_ops = tf.metrics.accuracy(
                    labels=tf.argmax(label, axis=-1),
                    predictions=tf.argmax(prob, axis=-1),
                    name='accuracy')
                metric_dict['accuracy'] = acc_up_ops
            elif em.strip() == 'acc_at_k':
                # top k accuracy:
                acc_k = metric_opt.get('acc_k', 5)
                eq = tf.to_float(
                    tf.math.in_top_k(predictions=prob,
                                     targets=tf.argmax(label, axis=-1),
                                     k=acc_k, ))
                acc_k_total = tf.get_variable(  # num of right predictions
                    name='make_metrics_dict/acc_k_total',
                    shape=[], dtype=tf.float32,
                    initializer=tf.initializers.constant(0.0, tf.float32))
                acc_k_count = tf.get_variable(  # num of examples
                    name='make_metrics_dict/acc_k_count',
                    shape=[], dtype=tf.float32,
                    initializer=tf.initializers.constant(0.0, tf.float32))
                acc_k_total_all = acc_k_total.assign_add(
                    tf.reduce_sum(eq), read_value=True)
                acc_k_count_all = acc_k_count.assign_add(
                    tf.squeeze(tf.to_float(tf.shape(eq)[0])), read_value=True)
                acc_top_k = tf.math.divide(acc_k_total_all, acc_k_count_all)
                metric_dict['accuracy_top_%d' % acc_k] = acc_top_k
            elif em.strip() == 'precision_at_k':
                prec_k = metric_opt.get('prec_k', 5)
                _, prec_up_ops = tf.metrics.precision_at_k(
                    labels=tf.argmax(label, axis=-1),
                    predictions=prob,
                    k=prec_k
                )
                metric_dict['precision_at_%d'%prec_k] = prec_up_ops
            else:
                raise Exception('Un-known metrics')
    return metric_dict


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  if FLAGS.cfg_file is None:
    raise ValueError('You must supply the cfg file !')

  cfg = _cfg_from_file(FLAGS.cfg_file)
  eval_cfg = cfg['eval']

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
      split=eval_cfg['eval_split'],
      cfg=eval_cfg['dataset_opt'],
      is_training=False,
      shuffle=False,
      num_epoch=1)

  #### build training dataset pipline #####
  im_batch, label_batch, _ = dataset_siamese.build_siamese_input_pipline(
      phase='eval',
      dataset=siamese_dataset,
      min_resize_value=cfg.get('min_resize_value', None),
      max_resize_value=cfg.get('max_resize_value', None),
      # eval cfgs:
      batch_size=1,
      shuffle=False,
      aug_opt=None,
      crop_size=cfg['corp_size'],
      dataset_spec=dataset_spec,
      drop_remainder=eval_cfg.get('drop_remainder', False))

  ##### get logits ####
  global_steps = tf.train.get_or_create_global_step()
  logits, _ = feature_extractor.extract_features(
      images=im_batch,
      num_classes=None,
      output_stride=cfg['output_stride'],
      global_pool=True,
      model_variant=cfg['model_variant'],
      weight_decay=0.0,
      dropout_keep_prob=1.0,
      regularize_depthwise=False,
      reuse=tf.AUTO_REUSE,
      is_training=False,
      fine_tune_batch_norm=False,
      cfg=cfg)

  #### build distance ####
  distance = build_distance_for_pairs_batch(
      features=logits,
      is_training=False,
      d_cfg=cfg['distance_config'],
      scope='compute_distance')
  
  with tf.name_scope('make_pred_to_one-hot'):
      if len(distance.shape) == 4:
          distance = tf.squeeze(distance, axis=[-3, -2])
      elif len(distance.shape) == 2:
          distance = tf.identity(distance)
      else:
          raise Exception('distance shape not right: %s' % (distance.shape))
      prob = tf.nn.sigmoid(x=distance)
      same_id_batch = tf.to_int64(tf.math.greater(prob, eval_cfg['same_id_threshold']))
      same_id_batch = tf.one_hot(tf.squeeze(same_id_batch, axis=-1),
                           2, on_value=1, off_value=0)

  #### convert siamese label to one hot format ###
  with tf.name_scope('conver_label_to_one-hot'):
      if len(label_batch.shape) == 1:
          label_batch = tf.one_hot(label_batch, 2, on_value=1, off_value=0)
      elif len(label_batch.shape) == 2:
          label_batch = tf.one_hot(
              tf.squeeze(label_batch, axis=-1), 2, on_value=1, off_value=0)
      else:
          raise Exception('check len(label_batch.shape), '
                          'expext 1 or 2, got %s' % (len(label_batch.shape)))

  ##### predictions #####
  metric_dict = make_metric_dict(
      prob=same_id_batch,
      label=label_batch,
      metric_opt=eval_cfg['metric_opt'],)

  #### add summaries ####
  # Add summaries for all eval metrics:
  for m in metric_dict:
      tf.summary.scalar(
          name='eval_results/%s' % m,
          tensor=metric_dict[m])

  # merge all summaries
  eval_output_dir = os.path.join(FLAGS.ckpt_dir, 'eval_'+eval_cfg['eval_split'])
  if not os.path.isdir(eval_output_dir):
      os.mkdir(eval_output_dir)
  merged_summaries = tf.summary.merge_all()
  summaries_writer = tf.summary.FileWriter(
      logdir=eval_output_dir,
      graph=tf.get_default_graph())

  #### set up session config ####
  # savers:
  exclude_scope = eval_cfg.get('exclude_scopes', None)
  if exclude_scope is None:
      exclude_scope = 'make_metrics_dict'
  else:
      exclude_scope = exclude_scope + ',make_metrics_dict'
  var_to_restore_list = _var_to_restore(exclude_scope)

  if FLAGS.restore_global_step:  # for logging a series of evaluaton
    var_to_restore_list.append(global_steps)
  if (not FLAGS.eval_once) and (not FLAGS.restore_global_step):
        tf.logging.info(
            'global_step not restored, new summaries will over write older ones')
  for v in var_to_restore_list:
      print(v.op.name)

  restor_saver = tf.train.Saver(
      var_list=var_to_restore_list)

  # session config:
  sess_cfg = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
  sess_cfg.gpu_options.allow_growth = True

  #### eval the model ####
  for ckpt_file in checkpoints_iterator(
          checkpoint_dir=FLAGS.ckpt_dir,
          min_interval_secs=FLAGS.min_interval_secs,
          timeout=FLAGS.ckpt_timeout_sec,
          timeout_fn=lambda: _time_out_fn(FLAGS.ckpt_timeout_sec)):
      with tf.Session(config=sess_cfg) as sess:
          tf.logging.info('Evaluating on ckpt: %s', ckpt_file)
          # init
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

          # restore from a ckpt
          restor_saver.restore(sess, ckpt_file)

          # eval and save a summary
          itrs = 0
          while True:
              try:
                  summaried, metriced = sess.run(
                      [merged_summaries, metric_dict])
                  itrs += 1
                  if itrs % 1000 == 0:
                      tf.logging.info(' ')
                      for m in metriced:
                          tf.logging.info('at %d iter: %s=%f', itrs, m, metriced[m])
              except tf.errors.OutOfRangeError:
                  # write summaries
                  gs = global_steps.eval()
                  if gs == 0:  # use global step in ckpt name
                      logging_step = os.path.basename(ckpt_file).split('.')[1]
                      logging_step = logging_step.strip()
                      if logging_step[-1] not in ['0','1','2','3','4','5','6','7','8','9']:
                          raise Exception('No global step in ckpt filename!')
                      logging_step = int(logging_step[5:])
                  else:
                      logging_step = gs
                  summaries_writer.add_summary(summaried, logging_step)
                  # print results:
                  for m in metriced:
                      tf.logging.info('%s:, %f', m, metriced[m])
                  break  # end of dataset, terminate

          #  break if just eval once
          if FLAGS.eval_once:
              tf.logging.info('Eval Once Done!')
              break

  print("End of Eval !!!")











if __name__ == '__main__':
  tf.app.run()
