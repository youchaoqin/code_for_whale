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
import pandas as pd

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
tf.app.flags.DEFINE_string(
    'ref_features_npy', None,
    'path for already computed reference features, in .npy form')
tf.app.flags.DEFINE_string(
    'dut_features_npy', None,
    'path for already computed dut features, in .npy form')
tf.app.flags.DEFINE_integer(
    'one_feature_long', 1792,
    'length of one feature, 1794 is for mobilenetV2_1.4'
)

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
  DEBUG = True
  tf_record_base = '/home/westwell/Desktop/dolores_storage/humpback_whale_identification/' \
                   'data/all/tfrecord_single_image/'
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

  #### get similarity probs of two features ####
  ref_features = tf.placeholder(
      tf.float32, shape=[None, FLAGS.one_feature_long], name='ref_features')
  dut_feature = tf.placeholder(
      tf.float32, shape=[1, FLAGS.one_feature_long], name='dut_features')
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
  # load pre-computed features of dut and ref
  all_ref_features_np = np.load(FLAGS.ref_features_npy)
  all_ref_features = all_ref_features_np[:, :-2]
  all_ref_image_names = all_ref_features_np[:, -2].tolist()
  all_ref_classes = all_ref_features_np[:, -1].tolist()

  all_dut_features_np = np.load(FLAGS.dut_features_npy)
  all_dut_features = all_dut_features_np[:, :-2]
  all_dut_image_names = all_dut_features_np[:, -2].tolist()
  all_dut_classes = all_dut_features_np[:, -1].tolist()

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

      # got prob_same_id for every test image and write result
      # submission file
      for nw_prob in FLAGS.new_whale_prob:
          # # a new submission file
          # output_file_path = os.path.join(
          #     FLAGS.output_dir, '..',
          #     'submission_%s_%s.csv'%(nw_prob, time.time()))
          # if os.path.isfile(output_file_path):
          #     raise Exception("submission file exists!! : %s" % (output_file_path))
          # with open(output_file_path, 'w') as f:
          #     f.write('Image,Id\n')

          for i in range(len(all_dut_image_names)):
              one_prob_same_ids = sess.run(
                  tf.get_default_graph().get_tensor_by_name(
                      'similarity_prob_for_one_query/prob_same_ids:0'),
                  feed_dict={'ref_features:0': all_ref_features,
                             'dut_features:0': np.expand_dims(all_dut_features[i],axis=0)})
              if not DEBUG:
                  one_prob_same_ids = np.concatenate(
                      (np.squeeze(one_prob_same_ids), [nw_prob]), axis=0)
              if i %100 == 0:
                  print('compare with: %f'%(nw_prob), i, all_dut_image_names[i],
                        one_prob_same_ids.min(), one_prob_same_ids.max())
              one_order = np.argsort(np.squeeze(one_prob_same_ids))[::-1]  # prob index
              one_order = one_order.tolist()
              if DEBUG:
                  orders_for_one_query = np.concatenate(
                      (all_ref_features_np[:, -2:], one_prob_same_ids),
                      axis=1)  # add probs and sort
                  # print(all_ref_features_np[:, -2:].shape)
                  # print(one_prob_same_ids.shape)
                  # print(orders_for_one_query.shape)
                  # print(one_order)
                  orders_for_one_query = orders_for_one_query[one_order, :]
                  tmp_pd = pd.DataFrame(
                      orders_for_one_query, columns=['im_name', 'im_cls', 'prob'])
                  csv_path = os.path.join(
                      FLAGS.output_dir,
                      '..',
                      'debug_results',
                      'result_%s---%s.csv' % (all_dut_image_names[i].decode()[:-4],
                                            all_dut_classes[i].decode()))
                  tmp_pd.to_csv(csv_path, index=False)

              # # make result for one iamges
              # one_predictions = []
              # for idx in one_order:
              #     tmp_prediction = all_ref_classes[idx]
              #     if tmp_prediction not in one_predictions:
              #         one_predictions.append(tmp_prediction)
              #     if len(one_predictions) == 5: # write one result
              #         with open(output_file_path, 'a') as f:
              #             content = os.path.basename(all_dut_image_names[i].decode()) + ','
              #             for j in range(len(one_predictions)):
              #                 if j == 0:
              #                     content = content + one_predictions[j].decode()
              #                 else:
              #                     content = content + ' ' + one_predictions[j].decode()
              #             content = content + '\n'
              #             f.write(content)
              #         break  # finish on dut image
              i += 1

if __name__ == '__main__':
  tf.app.run()
