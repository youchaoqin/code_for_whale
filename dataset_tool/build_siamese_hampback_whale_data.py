# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""convert tiny imagenet dataset into tfrecord
"""
import math
import os
import sys
import build_data
import tensorflow as tf
import collections
import six
import random
import numpy as np
import yaml
import pandas as pd
import copy

def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.
  Args:
    values: A scalar or list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def _convert_siamese_dataset(
        split, image_folder, annos, tfrecord_base_dir, num_shards=10):
    # shuffle annos
    sf_annos = copy.deepcopy(annos)
    random.shuffle(sf_annos)
    random.shuffle(sf_annos)
    random.shuffle(sf_annos)

    # convert each row in annos into tfrecord
    num_annos = len(sf_annos)
    num_per_shard = int(math.ceil(num_annos / float(num_shards)))

    image_reader = build_data.ImageReader('jpeg', channels=3)

    for shard_id in range(num_shards):
        output_filename = os.path.join(
            tfrecord_base_dir,
            '%s-%05d-of-%05d.tfrecord' % (split, shard_id, num_shards))
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_annos)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_annos, shard_id))
                sys.stdout.flush()

                # Read the image and extract more information
                id_a = sf_annos[i][0]
                id_b = sf_annos[i][1]
                im_a_name = sf_annos[i][2]
                im_a = tf.gfile.FastGFile(os.path.join(image_folder, im_a_name),
                                          'rb').read()
                im_a_h, im_a_w = image_reader.read_image_dims(im_a)
                im_b_name = sf_annos[i][3]
                im_b = tf.gfile.FastGFile(os.path.join(image_folder, im_b_name),
                                          'rb').read()
                im_b_h, im_b_w = image_reader.read_image_dims(im_b)
                label = int(sf_annos[i][4])

                # make one tf example
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'id_a': _bytes_list_feature(id_a),
                        'im_a': _bytes_list_feature(im_a),
                        'im_a_name': _bytes_list_feature(im_a_name),
                        'im_a_h': _int64_list_feature(im_a_h),
                        'im_a_w': _int64_list_feature(im_a_w),
                        'id_b': _bytes_list_feature(id_b),
                        'im_b': _bytes_list_feature(im_b),
                        'im_b_name': _bytes_list_feature(im_b_name),
                        'im_b_h': _int64_list_feature(im_b_h),
                        'im_b_w': _int64_list_feature(im_b_w),
                        'label': _int64_list_feature(label),
                    }))
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    base_dir = '/home/westwell/Desktop/dolores_storage/' \
               'humpback_whale_identification/data/all/'
    tfrecord_base_dir = os.path.join(base_dir, 'tfrecord_siamese_paires_10')
    if not os.path.isdir(tfrecord_base_dir):
        os.mkdir(tfrecord_base_dir)

    anno_files = [
                 #'siamese_pairs_5_tiny.csv',
                 'siamese_pairs_10_val.csv',
                 'siamese_pairs_10_train.csv'
                ]

    # conver each .csv file to tfrecord
    for anno_file in anno_files:
        # load anno_file and converte it to a list
        annos = pd.read_csv(os.path.join(base_dir, anno_file))
        annos = annos.values
        annos = annos.tolist()

        _convert_siamese_dataset(
            split=anno_file[:-4],
            image_folder=os.path.join(base_dir, 'train'),
            annos=annos,
            tfrecord_base_dir=tfrecord_base_dir,
            num_shards=10)


















