"""
functions and classes to build dataset with tfrecord and make augmentations
"""
import tensorflow as tf
import numpy as np
import os
from data_transformation import resize_to_range, corp_image, image_augmentation

SHUFFLE_BUFFER_SIZE = 10000

### parse_fn_map and parse_fn ###
# the parse_fn expect to returen decoded_image, index_label

def parser_tiny_image_net(record, phase='train'):
    with tf.name_scope('parse_tiny_imagenet'):
        features = tf.parse_single_example(
            serialized=record,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'bbox/xmin': tf.FixedLenFeature([], tf.int64),
                'bbox/ymin': tf.FixedLenFeature([], tf.int64),
                'bbox/xmax': tf.FixedLenFeature([], tf.int64),
                'bbox/ymax': tf.FixedLenFeature([], tf.int64),
            }
        )
        label = features['label']
        xmin = tf.cast(features['bbox/xmin'] if phase == 'train' else 0, tf.int32)
        ymin = tf.cast(features['bbox/ymin'] if phase == 'train' else 0, tf.int32)
        xmax = tf.cast(features['bbox/xmax'] if phase == 'train' else 63, tf.int32)
        ymax = tf.cast(features['bbox/ymax'] if phase == 'train' else 63, tf.int32)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.crop_to_bounding_box(
            image=image,
            offset_height=ymin, offset_width=xmin,
            target_height=ymax-ymin+1, target_width=xmax-xmin+1)
    return image, label

def parser_humpback_whale(record, phase='train'):
    with tf.name_scope('parser_humpback_whale'):
        features = tf.parse_single_example(
            serialized=record,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'image_name': tf.FixedLenFeature([], tf.string),
                'class_name': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.image.decode_jpeg(features['image'], channels=3)
        label = features['label']
    return image, label


parse_fn_map = {
    'tiny_imagenet': parser_tiny_image_net,
    'hampback_whale': parser_humpback_whale,
}



class Cls_Dataset(tf.data.TFRecordDataset):
    def __init__(self,
                 filenames, compression_type, buffer_size, num_parallel_reads,
                 dataset_name, num_classes, split_name, num_examples):
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
        self.num_classes = num_classes
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


def get_dataset(dataset_folder, split, cfg):
    """
    Args:
        :param dataset_folder: folder contain tfrecords for the
            the dataset, also contains a dataset_spec.yml describe
            some basic information such as num_cls, splits, num_exaples
            of different splits
        :param split: the split to build the dataset
        :param cfg: dataset config such as buffer_size
    Returns:
        :return: cls_dataset, a cls_dataset object
    """
    with tf.name_scope('DataSet'):
        # read the dataset_spec
        dataset_spec = _cfg_from_file(
            os.path.join(dataset_folder, 'dataset_spec.yml'))

        # read the split filenames
        split_filenames = _get_tfrecord_names(dataset_folder, split)

        # make the cls_dataset objects
        cls_dataset = Cls_Dataset(
            filenames=split_filenames,
            compression_type=cfg.get('compression_type', None),
            buffer_size=cfg.get('buffer_size', None),
            num_parallel_reads=cfg.get('num_parallel_reads', None),
            # dataset specification:
            dataset_name=dataset_spec['dataset_name'],
            num_classes=dataset_spec['num_classes'],
            split_name=split,
            num_examples=dataset_spec['num_examples'][split],)
    return cls_dataset


def build_input_pipline(phase, dataset, min_resize_value, max_resize_value, batch_size,
                        num_epoch, shuffle, aug_opt, crop_size):
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
            dset = dset.shuffle(shuffle_buffer_size)
        dset = dset.batch(batch_size).repeat(num_epoch)

        #### make iterator ####
        iterator = dset.make_one_shot_iterator()  # since we use abolute filenames
        im_batch, label_batch = iterator.get_next()

    print("########################## input pipline ##########################")
    return im_batch, label_batch




