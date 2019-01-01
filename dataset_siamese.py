"""
functions and classes to build dataset with tfrecord and make augmentations
"""
import tensorflow as tf
import numpy as np
import os
from data_transformation import resize_to_range, corp_image, image_augmentation

SHUFFLE_BUFFER_SIZE = 10000

### parse_fn_map and parse_fn ###
# the parse_fn expect to returen decoded_image_pairs, match_label
def parser_siamese_humpback_whale(record, phase='train'):
    with tf.name_scope('parser_siamese_humpback_whale'):
        features = tf.parse_single_example(
            serialized=record,
            features={
                #'id_a': tf.FixedLenFeature([], tf.string),
                'im_a': tf.FixedLenFeature([], tf.string),
                'im_a_name': tf.FixedLenFeature([], tf.string),
                #'im_a_h': tf.FixedLenFeature([], tf.int64),
                #'im_a_w': tf.FixedLenFeature([], tf.int64),
                #'id_b': tf.FixedLenFeature([], tf.string),
                'im_b': tf.FixedLenFeature([], tf.string),
                'im_b_name': tf.FixedLenFeature([], tf.string),
                #'im_b_h': tf.FixedLenFeature([], tf.int64),
                #'im_b_w': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )

        #id_a = features['id_a']
        im_a = tf.image.decode_image(features['im_a'], channels=3)
        im_a_name = features['im_a_name']
        #im_a_h = features['im_a_h']
        #im_a_w = features['im_a_w']

        #id_b = features['id_b']
        im_b = tf.image.decode_image(features['im_b'], channels=3)
        im_b_name = features['im_b_name']
        #im_b_h = features['im_b_h']
        #im_b_w = features['im_b_w']

        label = features['label']

    return im_a, im_b, label, im_a_name, im_b_name

parse_fn_map = {
    'hampback_whale_siamese': parser_siamese_humpback_whale,
}


class Siamese_Dataset(tf.data.TFRecordDataset):
    def __init__(self,
                 filenames, compression_type, buffer_size, num_parallel_reads,
                 dataset_name, split_name, num_examples):
        if num_parallel_reads is not None:
            if num_parallel_reads > len(filenames):
                _parallel_readers = len(filenames)
            else:
                _parallel_readers = num_parallel_reads
        else:
            _parallel_readers = len(filenames)
        print('##### num_parallel_reads: %d #####' % (_parallel_readers))
        super().__init__(filenames=filenames, compression_type=compression_type,
                         buffer_size=buffer_size, num_parallel_reads=_parallel_readers)
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.num_examples = num_examples


def _cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def _get_tfrecord_names(folder, split):
    tfrecord_files = []
    files_list = os.listdir(folder)
    for f in files_list:
        if (split in f) and ('.tfrecord' in f):
            tfrecord_files.append(os.path.join(folder, f))
    return tfrecord_files


def get_siamese_dataset(dataset_folder, split, cfg, is_training=True, shuffle=True,
                        num_epoch=1):
    """
    Args:
        :param dataset_folder: folder contain tfrecords for the
            the dataset
        :param split: the split to build the dataset
        :param cfg: dataset config such as buffer_size
        :param is_training: is training?
    Returns:
        :return: cls_dataset, a cls_dataset object
    """
    with tf.name_scope('DataSet'):
        # read the split filenames
        split_filenames = _get_tfrecord_names(dataset_folder, split)
        num_shards = len(split_filenames)

        # make the siamese_dataset objects
        filename_dataset = tf.data.Dataset.from_tensor_slices(split_filenames)

        filename_dataset = filename_dataset.repeat(num_epoch if is_training else 1)
        if is_training and shuffle:
            filename_dataset = filename_dataset.shuffle(buffer_size=num_shards)

        siamese_dataset = filename_dataset.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(
                filenames=filename,
                compression_type=cfg.get('compression_type', None),
                buffer_size=cfg.get('buffer_size', None),),
            cycle_length=int(num_shards/3)))
    return siamese_dataset


def build_siamese_input_pipline(
        phase, dataset, min_resize_value, max_resize_value, batch_size, shuffle,
        aug_opt, crop_size, dataset_spec, drop_remainder=False):
    print("########################## input pipline ##########################")
    with tf.name_scope('Siamese_Input_Pipline'):
        # parse dataset
        dset = dataset.map(
            lambda record: parse_fn_map[dataset_spec['dataset_name']](record, phase))

        # resize image size acrroding to min_resize_value and max_resize_value
        # and convert label to one-hot encode
        dset = dset.map(lambda im_a, im_b, label, im_a_name, im_b_name:\
                            (resize_to_range(im_a, min_resize_value,max_resize_value),
                             resize_to_range(im_b, min_resize_value, max_resize_value),
                             label, im_a_name, im_b_name))

        # do data augmentation
        if (aug_opt is not None) and (phase == 'train'):
            dset = dset.map(lambda im_a, im_b, label, im_a_name, im_b_name:\
                                (image_augmentation(im_a, aug_opt),
                                 image_augmentation(im_b, aug_opt),
                                 label, im_a_name, im_b_name))

        # corp to crop_size for training or evaluation
        if phase == 'train' and (aug_opt is not None):
            random_crop = aug_opt.get('random_crop', False)
            assert isinstance(random_crop, bool)
        else:
            random_crop = False
        if random_crop:
            print('### apply random crop with %s ###'%(crop_size))
        else:
            print('### crop with %s ###' % (crop_size))
        if crop_size is None:
            if batch_size != 1:
                raise Exception('crop_size should not be None for batch_size > 1')
            else:
                pass
        else:
            dset = dset.map(lambda im_a, im_b, label, im_a_name, im_b_name:\
                                (corp_image(im_a, crop_size, random_crop=random_crop),
                                 corp_image(im_b, crop_size, random_crop=random_crop),
                                 label, im_a_name, im_b_name))
        # shuffle and batch
        if shuffle:
            if aug_opt is not None:
                shuffle_buffer_size = aug_opt.get(
                    'shuffle_buffer_size', SHUFFLE_BUFFER_SIZE)
            else:
                shuffle_buffer_size = SHUFFLE_BUFFER_SIZE
            print('### shuffle with buffer_size: %d ###' % (shuffle_buffer_size))
            dset = dset.prefetch(8*batch_size)
            dset = dset.shuffle(shuffle_buffer_size)
        dset = dset.batch(batch_size, drop_remainder=drop_remainder)
        dset = dset.prefetch(8)

        #### make iterator ####
        iterator = dset.make_one_shot_iterator()  # since we use absolute filenames
        im_a_batch, im_b_batch, label_batch, \
        im_a_name_batch, im_b_name_batch = iterator.get_next()

        im_batch = tf.concat([im_a_batch, im_b_batch], axis=0, name='im_a_b_concat')
        im_name_batch = tf.concat([im_a_name_batch, im_b_name_batch], axis=0,
                                  name='ia_a_b_name_cancat')

    print("########################## input pipline ##########################")
    return im_batch, label_batch, im_name_batch



def build_input_pipline(phase, dataset, min_resize_value, max_resize_value, batch_size,
                        num_epoch, shuffle, aug_opt, crop_size, drop_remainder=False):
    print("########################## input pipline ##########################")
    with tf.name_scope('Input_Pipline'):
        # parse dataset
        dset = dataset.map(
            lambda record: parse_fn_map[dataset.dataset_name](record, phase))

        # resize image size acrroding to min_resize_value and max_resize_value
        # and convert label to one-hot encode
        dset = dset.map(lambda im, label: (resize_to_range(im, min_resize_value,
                                                           max_resize_value),
                                           tf.one_hot(label, dataset.num_classes)))

        # do data augmentation
        if (aug_opt is not None) and (phase == 'train'):
            dset = dset.map(lambda im, label: (image_augmentation(im, aug_opt),
                                               label))

        # corp to crop_size for training or evaluation
        if phase == 'train' and (aug_opt is not None):
            random_crop = aug_opt.get('random_crop', False)
            assert isinstance(random_crop, bool)
        else:
            random_crop = False
        if random_crop:
            print('### apply random crop with %s ###'%(crop_size))
        else:
            print('### crop with %s ###' % (crop_size))
        if crop_size is None:
            if batch_size != 1:
                raise Exception('crop_size should not be None for batch_size > 1')
            else:
                pass
        else:
            dset = dset.map(lambda im, label: (corp_image(im, crop_size,
                                                          random_crop=random_crop),
                                                label))
        # shuffle and batch
        if shuffle:
            if aug_opt is not None:
                shuffle_buffer_size = aug_opt.get(
                    'shuffle_buffer_size', SHUFFLE_BUFFER_SIZE)
            else:
                shuffle_buffer_size = SHUFFLE_BUFFER_SIZE
            print('### shuffle with buffer_size: %d ###' % (shuffle_buffer_size))
            dset = dset.prefetch(8*batch_size)
            dset = dset.shuffle(shuffle_buffer_size)
        dset = dset.batch(batch_size, drop_remainder=drop_remainder).repeat(num_epoch)
        dset = dset.prefetch(8)

        #### make iterator ####
        iterator = dset.make_one_shot_iterator()  # since we use abolute filenames
        im_batch, label_batch = iterator.get_next()

    print("########################## input pipline ##########################")
    return im_batch, label_batch




