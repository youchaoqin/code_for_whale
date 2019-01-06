"""train a specific model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dataset
import math
import numpy as np
import cv2
import os
import feature_extractor
import time

from data_transformation import resize_to_range, corp_image
from build_distance import similarity_prob_for_one_query

slim = tf.contrib.slim

##### train configs #####
tf.app.flags.DEFINE_string('gpu', '0', 'CUDA_VISIBLE_DEVICES')
tf.app.flags.DEFINE_string(
    'cfg_file', None,
    'cfg file path, cfg file contains paremeters for training')
tf.app.flags.DEFINE_string(
    'output_dir', None,
    'output dir to save ckpts and summaries.')
tf.app.flags.DEFINE_multi_float(
    'new_whale_prob', [0.5, 0.4, 0.3, 0.2, 0.1], 'prob of new_whale')

FLAGS = tf.app.flags.FLAGS
#########################
#########################

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


def siamese_prob(ref_features, dut_feature, distance_type, scope='siamese_distance'):
    if distance_type == 'weighted_l1_distance':
        alpha = tf.get_variable(
            name='siamese_distance/l1_alpha',
            shape=[1, ref_features.shape[-1]], dtype=tf.float32,
            initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.2),
            trainable=False,collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                         tf.GraphKeys.MODEL_VARIABLES])
        distances = tf.abs(tf.subtract(ref_features, dut_feature))
        distances = tf.reduce_sum(tf.multiply(distances, alpha), axis=-1)
        prob_same_ids = tf.nn.sigmoid(distances, name='prob_same_ids')
    else:
        raise Exception('distance not impelemented yet %s' % (distance_type))

    return prob_same_ids


def _parser_humpback_whale(record, phase='train'):
    with tf.name_scope('parser_humpback_whale'):
        features = tf.parse_single_example(
            serialized=record,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_name': tf.FixedLenFeature([], tf.string),
                'class_name': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.image.decode_jpeg(features['image'], channels=3)
        label = features['label']
        image_name = features['image_name']
        class_name = features['class_name']
        height = features['height']
        width = features['width']
    return image, label, image_name, class_name, height, width


def _get_tfrecord_names(folder, split):
    tfrecord_files = []
    files_list = os.listdir(folder)
    for f in files_list:
        if (split in f) and ('.tfrecord' in f):
            tfrecord_files.append(os.path.join(folder, f))
    return tfrecord_files


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
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

  #### get features ####
  input_image = tf.placeholder(tf.uint8, shape=[None, None, 3], name='input_image')
  image = resize_to_range(input_image, cfg['min_resize_value'], cfg['max_resize_value'])
  image = corp_image(image, cfg['corp_size'], random_crop=False)
  image = tf.expand_dims(image, axis=0)
  feature_for_dst, _ = feature_extractor.extract_features(
      images=image,
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
  if len(feature_for_dst.shape) == 4:
      feature_for_dst = tf.squeeze(
          feature_for_dst, axis=[1,2], name='features_for_dst')
  elif len(feature_for_dst.shape) == 2:
      feature_for_dst = tf.identity(feature_for_dst, name='features_for_dst')
  else:
      raise Exception('feature_for_dst shape not right, got %s'%(feature_for_dst.shape))

  #### get similarity probs of two features ####
  ref_features = tf.placeholder(
      tf.float32, shape=[None, feature_for_dst.shape[-1]], name='ref_features')
  dut_feature = tf.placeholder(
      tf.float32, shape=[1, feature_for_dst.shape[-1]], name='dut_features')
  prob_same_ids = similarity_prob_for_one_query(
      ref_features=ref_features,
      dut_feature=dut_feature,
      d_cfg=cfg['distance_config'],
      scope='similarity_prob_for_one_query')

  #### set up session config ####
  # session config:
  sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  sess_cfg.gpu_options.allow_growth = True

  #### do test the model ####
  with tf.Session(config=sess_cfg) as sess:
      # init
      #sess.run(tf.global_variables_initializer())
      #sess.run(tf.local_variables_initializer())

      # restore vars from pretrained ckpt:
      vars_to_restore = _var_to_restore(None)
      for v in vars_to_restore:
          print(v.op.name)
      restor_saver = tf.train.Saver(var_list=vars_to_restore)
      restor_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))

      # forward all ref images
      filenames = _get_tfrecord_names(
          '/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
          'data/all/tfrecord_no_new_whale/', 'train_no_new_whale')
      dataset = tf.data.TFRecordDataset(filenames)
      dataset = dataset.map(lambda record: _parser_humpback_whale(record, 'eval'))
      dataset.batch(batch_size=1)
      iterator = dataset.make_one_shot_iterator()
      ref_image, _, _, ref_class_name, _, _ = iterator.get_next()

      all_ref_features = None
      all_ref_cls_name = []
      i = 0
      while True:
          try:
              one_ref_image, one_ref_class_name = sess.run([ref_image, ref_class_name])
              if i % 100 == 0:
                  print(i, one_ref_class_name)
              all_ref_cls_name.append(one_ref_class_name)
              one_ref_feature = sess.run(
                  tf.get_default_graph().get_tensor_by_name('features_for_dst:0'),
                  feed_dict={'input_image:0': one_ref_image})
              if all_ref_features is None:
                  all_ref_features = one_ref_feature
              else:
                  all_ref_features = np.concatenate(
                      (all_ref_features, one_ref_feature), axis=0)
              i += 1
          except tf.errors.OutOfRangeError:
              tf.logging.info('End of forward ref images')
              break
      all_ref_cls_name.append('new_whale'.encode(encoding='utf-8'))

      # forward all test images
      filenames = _get_tfrecord_names(
          '/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
          'data/all/tfrecord_no_new_whale/', 'test')
      dataset = tf.data.TFRecordDataset(filenames)
      dataset = dataset.map(lambda record: _parser_humpback_whale(record, 'eval'))
      dataset.batch(batch_size=1)
      iterator = dataset.make_one_shot_iterator()
      dut_image, _, dut_image_name, _, _, _ = iterator.get_next()

      all_dut_featurs = None
      all_dut_image_names = []
      i = 0
      while True:
          try:
              one_dut_image, one_dut_image_name = sess.run([dut_image, dut_image_name])
              if i % 100 == 0:
                  print(i, one_dut_image_name)
              all_dut_image_names.append(one_dut_image_name)
              one_dut_feature = sess.run(
                  tf.get_default_graph().get_tensor_by_name('features_for_dst:0'),
                  feed_dict={'input_image:0': one_dut_image})
              if all_dut_featurs is None:
                  all_dut_featurs = one_dut_feature
              else:
                  all_dut_featurs = np.concatenate(
                      (all_dut_featurs, one_dut_feature), axis=0)
              i += 1
          except tf.errors.OutOfRangeError:
              tf.logging.info('End of forward dut images')
              break

      # got prob_same_id for every test image and write result
      # submission file
      for nw_prob in FLAGS.new_whale_prob:
          output_file_path = os.path.join(
              FLAGS.output_dir, '..',
              'submission_%s_%s.csv'%(nw_prob, time.time()))
          if os.path.isfile(output_file_path):
              raise Exception("submission file exists!! : %s" % (output_file_path))
          with open(output_file_path, 'w') as f:
              f.write('Image,Id\n')

          for i in range(len(all_dut_image_names)):
              if i %100 == 0:
                  print('compare with: %f'%(nw_prob), i, all_dut_image_names[i])

              one_prob_same_ids = sess.run(
                  tf.get_default_graph().get_tensor_by_name(
                      'similarity_prob_for_one_query/prob_same_ids:0'),
                  feed_dict={'ref_features:0': all_ref_features,
                             'dut_features:0': np.expand_dims(all_dut_featurs[i],axis=0)})
              one_prob_same_ids = np.concatenate(
                  (one_prob_same_ids, [nw_prob]), axis=0)
              one_order = np.argsort(one_prob_same_ids)[::-1]  # prob index
              one_order = one_order.tolist()

              one_predictions = []
              for idx in one_order:
                  tmp_prediction = all_ref_cls_name[idx]
                  if tmp_prediction not in one_predictions:
                      one_predictions.append(tmp_prediction)
                  if len(one_predictions) == 5: # write one result
                      with open(output_file_path, 'a') as f:
                          content = os.path.basename(all_dut_image_names[i].decode()) + ','
                          for j in range(len(one_predictions)):
                              if j == 0:
                                  content = content + one_predictions[j].decode()
                              else:
                                  content = content + ' ' + one_predictions[j].decode()
                          content = content + '\n'
                          f.write(content)
                      break  # finish on dut image
              i += 1

if __name__ == '__main__':
  tf.app.run()
