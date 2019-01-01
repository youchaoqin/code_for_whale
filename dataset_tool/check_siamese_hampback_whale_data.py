import tensorflow as tf
import numpy as np
import os
import yaml
import cv2
import matplotlib.pylab as plt


def parser_siamese_humpback_whale(record, phase='train'):
    with tf.name_scope('parser_siamese_humpback_whale'):
        features = tf.parse_single_example(
            serialized=record,
            features={
                'id_a': tf.FixedLenFeature([], tf.string),
                'im_a': tf.FixedLenFeature([], tf.string),
                'im_a_name': tf.FixedLenFeature([], tf.string),
                'im_a_h': tf.FixedLenFeature([], tf.int64),
                'im_a_w': tf.FixedLenFeature([], tf.int64),
                'id_b': tf.FixedLenFeature([], tf.string),
                'im_b': tf.FixedLenFeature([], tf.string),
                'im_b_name': tf.FixedLenFeature([], tf.string),
                'im_b_h': tf.FixedLenFeature([], tf.int64),
                'im_b_w': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )

        id_a = features['id_a']
        im_a = tf.image.decode_image(features['im_a'], channels=3)
        im_a_name = features['im_a_name']
        im_a_h = features['im_a_h']
        im_a_w = features['im_a_w']

        id_b = features['id_b']
        im_b = tf.image.decode_image(features['im_b'], channels=3)
        im_b_name = features['im_b_name']
        im_b_h = features['im_b_h']
        im_b_w = features['im_b_w']

        label = features['label']

    return id_a, im_a, im_a_name, im_a_h, im_a_w, \
           id_b, im_b, im_b_name, im_b_h, im_b_w, \
           label


def _get_tfrecord_names(folder, split):
    tfrecord_files = []
    files_list = os.listdir(folder)
    for f in files_list:
        if (split in f) and ('.tfrecord' in f):
            tfrecord_files.append(os.path.join(folder, f))
    return tfrecord_files

if __name__ == '__main__':
    show_every = 10000
    tfrecord_folder = '/home/ycq/Desktop/humpback_whale_identification/data/all/' \
                      'tfrecord_siamese_paires_5'
    split_to_check = [
        'siamese_pairs_5_all',
    ]

    with open(os.path.join(tfrecord_folder, 'dataset_spec.yml'), 'r') as f:
        dataset_spec = yaml.load(f)

    for s in split_to_check:
        filenames = _get_tfrecord_names(tfrecord_folder, s)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda record: parser_siamese_humpback_whale(record, s))
        dataset.batch(batch_size=1)
        iterator = dataset.make_one_shot_iterator()
        id_a, im_a, im_a_name, im_a_h, im_a_w, \
        id_b, im_b, im_b_name, im_b_h, im_b_w, \
        label = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for i in range(dataset_spec['num_examples'][s]):
                one_id_a, one_im_a, one_im_a_name, one_im_a_h, one_im_a_w, \
                one_id_b, one_im_b, one_im_b_name, one_im_b_h, one_im_b_w, \
                one_label = sess.run(
                    [id_a, im_a, im_a_name, im_a_h, im_a_w,
                     id_b, im_b, im_b_name, im_b_h, im_b_w,
                     label])
                if i % show_every == 0:
                    # image a
                    plt.subplot(2, 1, 1)
                    plt.imshow(one_im_a)
                    plt.title(one_im_a_name.decode()+"_"+one_id_a.decode()+"_"+str(one_im_a_h)+\
                              "_"+str(one_im_a_w)+"_"+str(one_label))
                    # image b
                    plt.subplot(2, 1, 2)
                    plt.imshow(one_im_b)
                    plt.title(one_im_b_name.decode() + "_" + one_id_b.decode() + "_" + str(one_im_b_h) + \
                              "_" + str(one_im_b_w) + "_" + str(one_label))
                    plt.show()

