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


def _convert_dataset(split, image_folder, anno_file,
                     cls_name_to_index, tfrecord_base_dir, num_shards):
    split_statistics = {}
    # get raw information accroding to anno_file, and shuffle it
    anno_raw = []
    if anno_file is not None:
        with open(anno_file, 'r') as f:
            for l in f.readlines():
                if "Image" in l:
                    continue
                anno_temp = l.strip().split(',')
                im_name = anno_temp[0].strip()
                im_name = os.path.join(image_folder, im_name)  # to absolute addrs
                im_cls = anno_temp[1].strip()
                im_idx = cls_name_to_index[im_cls]
                anno_raw.append([im_name, im_cls, im_idx])
        random.shuffle(anno_raw)
        random.shuffle(anno_raw)
        random.shuffle(anno_raw)
    else:  # test set
        image_file_list = os.listdir(image_folder)
        for one_im in image_file_list:
            if one_im[-4:] == '.jpg':
                im_name = os.path.join(image_folder, one_im)
                im_cls = 'test_image'
                im_idx = 999999
                anno_raw.append([im_name, im_cls, im_idx])

    # extract all information of one image and write it tfrecords
    num_images = len(anno_raw)
    num_per_shard = int(math.ceil(num_images / float(num_shards)))

    image_reader = build_data.ImageReader('jpeg', channels=3)

    for shard_id in range(num_shards):
        output_filename = os.path.join(
            tfrecord_base_dir,
            '%s-%05d-of-%05d.tfrecord' % (split, shard_id, num_shards))
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image and extract more information
                image_data = tf.gfile.FastGFile(anno_raw[i][0], 'rb').read()
                image_name = os.path.basename(anno_raw[i][0])
                image_label = anno_raw[i][1]
                image_index = anno_raw[i][2]
                image_height, image_width = \
                    image_reader.read_image_dims(image_data)

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'image': _bytes_list_feature(image_data),
                        'image_name': _bytes_list_feature(image_name),
                        'class_name': _bytes_list_feature(image_label),
                        'label': _int64_list_feature(image_index),
                        'height': _int64_list_feature(image_height),
                        'width': _int64_list_feature(image_width),
                    })
                )
                tfrecord_writer.write(example.SerializeToString())
                anno_raw[i].extend([image_height,image_width])  # for statistic
        sys.stdout.write('\n')
        sys.stdout.flush()

    # for split_statistics
    split_statistics['split_name'] = split
    split_statistics['total_examples'] = len(anno_raw)
    height_all = []
    width_all = []
    split_statistics['per_class_num'] = {}
    for anno in anno_raw:
        one_cls = anno[1]
        height_all.append(anno[3])
        width_all.append(anno[4])
        if one_cls not in split_statistics['per_class_num']:
            split_statistics['per_class_num'][one_cls] = 1
        else:
            split_statistics['per_class_num'][one_cls] += 1
    split_statistics['height_mean'] = float(np.mean(height_all))
    split_statistics['height_std'] = float(np.std(height_all))
    split_statistics['width_mean'] = float(np.mean(width_all))
    split_statistics['width_std'] = float(np.std(width_all))

    return split_statistics


def _load_clsname_to_index_map(filenname):
    map_dict = {}
    with open(filenname, 'r') as f:
        i = 1
        for l in f.readlines():
            one_map = l.strip().split(':')
            one_cls = one_map[0].strip()
            one_index = int(one_map[1].strip())
            if one_cls not in map_dict:
                map_dict[one_cls] = one_index
            else:
                raise Exception(
                    'duplicated pair %s:%d at line:%d'%(one_cls, one_index, i))
            i += 1
    return map_dict

if __name__ == '__main__':
    base_dir = '/home/westwell/Desktop/dolores_storage/' \
               'humpback_whale_identification/data/all/'
    anno_file = 'train_no_new_whale.csv'
    dataset_splits = [#'train',
                      #'test',
                      'train_no_new_whale',
                      ]
    tfrecord_base_dir = os.path.join(base_dir, 'tfrecord_no_new_whale')
    if not os.path.isdir(tfrecord_base_dir):
        os.mkdir(tfrecord_base_dir)

    # load the cls_name to index table and conver it to dict
    cls_name_to_index = _load_clsname_to_index_map(
        os.path.join(base_dir, 'cls_name_to_index.txt'))

    # conver each split to tfrecord
    for s in dataset_splits:
        if 'train' in s:
            set_statistics = _convert_dataset(
                split=s,
                image_folder=os.path.join(base_dir, 'train'),
                anno_file=os.path.join(base_dir, anno_file),
                cls_name_to_index=cls_name_to_index,
                tfrecord_base_dir=tfrecord_base_dir,
                num_shards=10)
            with open(os.path.join(base_dir, '%s_set_statistics.yml' % (s)), 'a') as f:
                yaml.dump(set_statistics, f)
        elif 'test' in s:
            set_statistics = _convert_dataset(
                split=s,
                image_folder=os.path.join(base_dir, 'test'),
                anno_file=None,
                cls_name_to_index=cls_name_to_index,
                tfrecord_base_dir=tfrecord_base_dir,
                num_shards=10)
            with open(os.path.join(base_dir, '%s_set_statistics.yml' % (s)), 'a') as f:
                yaml.dump(set_statistics, f)
        else:
            raise Exception('Un-know split %s' % (s))

    # np.save(os.path.join(base_dir, 'train_set_statistics.npy'),
    #         train_set_statistics)
    # np.save(os.path.join(base_dir, 'test_set_statistics.npy'),
    #         test_set_statistics)




    # print(train_set_statistics['height_mean'])
    # print(train_set_statistics['height_std'])
    # print(train_set_statistics['width_mean'])
    # print(train_set_statistics['width_std'])


















