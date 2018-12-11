"""eval a specific model using a given dataset.
either eval once or eval repeatedly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.training.python.training.evaluation import checkpoints_iterator

import dataset
import math
import numpy as np
import cv2
import os
import feature_extractor
from build_loss import build_loss
from data_transformation import resize_to_range, corp_image


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
tf.app.flags.DEFINE_integer(
    'top_k', 5,
    'export_top_k results')
tf.app.flags.DEFINE_string(
    'output_dir', None,
    'the dir to output the submission file')
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
                # top k accuracy:
                acc_k = metric_opt.get('acc_k', 5)
                eq = tf.to_float(
                    tf.math.in_top_k(predictions=prob,
                                     targets=tf.argmax(label, axis=-1),
                                     k=acc_k,))
                acc_k_total = tf.get_variable( # num of right predictions
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

                metric_dict['accuracy'] = acc_up_ops
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
  idx_to_cls_map = {}
  with open('/home/westwell/Desktop/dolores_storage/'
            'humpback_whale_identification/data/all/index_to_cls_name.txt', 'r') as f:
      for l in f.readlines():
          one_line = l.strip().split(':')
          one_idx = str(one_line[0].strip())
          one_name = one_line[1].strip()
          idx_to_cls_map[one_idx] = one_name
          print(one_idx, idx_to_cls_map[one_idx])

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  if FLAGS.cfg_file is None:
    raise ValueError('You must supply the cfg file !')

  cfg = _cfg_from_file(FLAGS.cfg_file)

  # print all configs
  print('############################ cfg ############################')
  for k in cfg:
      print('%s: %s'%(k, cfg[k]))

  tf.logging.set_verbosity(tf.logging.INFO)
  #######################################################################
  ##############              sigle GPU version            ##############
  #######################################################################

  input_image = tf.placeholder(tf.uint8, shape=[None,  None, 3], )
  image = resize_to_range(input_image, cfg['min_resize_value'], cfg['max_resize_value'])
  image = corp_image(image, cfg['corp_size'], random_crop=False)
  image = tf.expand_dims(image, axis=0)

  ##### get probabilities ####
  global_steps = tf.train.get_or_create_global_step()
  logits, _ = feature_extractor.extract_features(
      images=image,
      num_classes=5005,
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
  with tf.name_scope('make_probabilities'):
      if len(logits.shape) == 4:
          logits = tf.squeeze(logits, axis=[-3, -2])
      elif len(logits.shape) == 2:
          logits = tf.identity(logits)
      else:
          raise Exception('logits shape not right: %s' % (logits.shape))
      prob = tf.nn.softmax(
          logits=logits, axis=-1, name='probabilities')

  _, top_k_idx = tf.nn.top_k(prob, k=FLAGS.top_k)
  top_k_idx = tf.squeeze(top_k_idx)

  #### set up session config ####
  # savers:
  var_to_restore_list = _var_to_restore('make_metrics_dict')

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
  # submission file
  output_file_path = os.path.join(FLAGS.output_dir, 'submission.csv')
  if os.path.isfile(output_file_path):
      raise Exception("submission file exists!! : %s" % (output_file_path))
  with open(output_file_path, 'r') as f:
      f.write('Image,Id\n')

  all_files = os.listdir('/home/westwell/Desktop/dolores_storage/'
                          'humpback_whale_identification/data/all/test/')
  all_images = []
  for f in all_files:
      if f[-4:] == '.jpg':
          all_images.append(os.path.join('/home/westwell/Desktop/dolores_storage/'
                                         'humpback_whale_identification/data/all/test/',
                                         f))

  for ckpt_file in checkpoints_iterator(
          checkpoint_dir=FLAGS.ckpt_dir,
          min_interval_secs=FLAGS.min_interval_secs,
          timeout=FLAGS.ckpt_timeout_sec,
          timeout_fn=lambda: _time_out_fn(FLAGS.ckpt_timeout_sec)):
      with tf.Session(config=sess_cfg) as sess:
          # init
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

          # restore from a ckpt
          restor_saver.restore(sess, ckpt_file)

          # test
          for i, im_file_path in enumerate(all_images):
              print(i, im_file_path)
              one_im = cv2.imread(im_file_path)
              one_im = cv2.cvtColor(one_im, cv2.COLOR_BGR2RGB)
              one_top_k_idx = sess.run(top_k_idx,
                                       feed_dict={input_image: one_im})
              one_top_k_idx = np.squeeze(one_top_k_idx)
              one_top_k_idx = one_top_k_idx.tolist()
              with open(output_file_path, 'a') as f:
                  content = os.path.basename(im_file_path)+','
                  for j, idx in enumerate(one_top_k_idx):
                      if j == 0:
                          content = content + idx_to_cls_map[str(idx)]
                      else:
                          content = content + ' ' + idx_to_cls_map[str(idx)]
                  content = content + '\n'
                  f.write(content)
      break

  print("End of Eval !!!")











if __name__ == '__main__':
  tf.app.run()
