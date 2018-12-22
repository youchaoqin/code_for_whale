import tensorflow as tf
import numpy as np
import os
import yaml
import cv2

def parser_humpback_whale(record, phase='train'):
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

if __name__ == '__main__':
    tfrecord_folder = '/home/westwell/Desktop/dolores_storage/' \
                      'humpback_whale_identification/data/all/tfrecord_no_new_whale/'
    split_to_check = [#'train',
                      #'val',
                      #'test',
                      'train_no_new_whale',
                    ]
    with open(os.path.join(tfrecord_folder, 'dataset_spec.yml'), 'r') as f:
        dataset_spec = yaml.load(f)

    for s in split_to_check:
        filenames = _get_tfrecord_names(tfrecord_folder, s)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda record: parser_humpback_whale(record, s))
        dataset.batch(batch_size=1)
        iterator = dataset.make_one_shot_iterator()
        image, label, image_name, class_name, height, width = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for i in range(dataset_spec['num_examples'][s]):
                image_t, label_t, image_name_t, \
                class_name_t, height_t, width_t = sess.run([image, label, image_name,
                                                            class_name, height, width])
                if i % 1000 == 0:
                    print(label_t, image_name_t, class_name_t, height_t, width_t)
                    image_t = cv2.cvtColor(image_t, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(
                            tfrecord_folder, 'dataset_test',
                                             str(i)+'_'+str(image_name_t[:-4])+'_'+str(class_name_t)+ \
                                             '_'+str(label_t)+'_'+str(height_t)+'x'+ \
                                             str(width_t)+'_.jpg'), image_t)
