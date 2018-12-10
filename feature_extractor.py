"""Extracts features for different models.
borrowed a lot of code from deeplab
"""
import functools
import tensorflow as tf

from nets.resnet_v2 import resnet_v2_50
from nets.resnet_utils import resnet_arg_scope
import nets.mobilenet.mobilenet_v2 as mobilenet_v2

slim = tf.contrib.slim

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  return inputs - mean_rgb


def _preprocess_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0


def mean_pixel(model_variant=None):
  """Gets mean pixel value.

  This function returns different mean pixel value, depending on the input
  model_variant which adopts different preprocessing functions. We currently
  handle the following preprocessing functions:
  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.

  Args:
    model_variant: Model variant (string) for feature extraction. For
      backwards compatibility, model_variant=None returns _MEAN_RGB.

  Returns:
    Mean pixel value.
  """
  if model_variant in ['resnet_v1_50',
                       'resnet_v1_101'] or model_variant is None:
    return _MEAN_RGB
  else:
    return [127.5, 127.5, 127.5]

##########################################################################
######################## Registe the Net #################################
##########################################################################

### A map from network name to network function.
networks_map = {
    'resnet_v2_50': resnet_v2_50,
    'mobilenet_v2_1.4': mobilenet_v2.mobilenet,
    'mobilenet_v2_1.0': mobilenet_v2.mobilenet,
}

### A map from network name to network arg scope.
arg_scopes_map = {
    'resnet_v2_50': resnet_arg_scope,
    'mobilenet_v2_1.4': mobilenet_v2.training_scope,
    'mobilenet_v2_1.0': mobilenet_v2.training_scope
}

### A map from feature extractor name to the network name scope used in the
### ImageNet pretrained versions of these models.
name_scope = {
    'resnet_v2_50': 'resnet_v2_50',
    'mobilenet_v2_1.4': 'MobilenetV2',
    'mobilenet_v2_1.0': 'MobilenetV2',
}

### A map from feature extractor name to the network preprocessing function
_PREPROCESS_FN = {
    'resnet_v2_50': _preprocess_zero_mean_unit_range,
    'mobilenet_v2_1.4': _preprocess_zero_mean_unit_range,
    'mobilenet_v2_1.0': _preprocess_zero_mean_unit_range,
}

##########################################################################
##########################################################################

def get_network(network_name, preprocess_images, arg_scope=None):
  """Gets the network.

  Args:
    network_name: Network name.
    preprocess_images: Preprocesses the images or not.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.

  Returns:
    A network function that is used to extract features.

  Raises:
    ValueError: network is not supported.
  """
  if network_name not in networks_map:
    raise ValueError('Unsupported network %s.' % network_name)
  arg_scope = arg_scope or arg_scopes_map[network_name]()
  def _identity_function(inputs):
    return inputs
  if preprocess_images:
    preprocess_function = _PREPROCESS_FN[network_name]
  else:
    preprocess_function = _identity_function
  func = networks_map[network_name]
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      preprocessed_im = tf.identity(preprocess_function(inputs),
                                    name='EncoderInputTensor')
      return func(preprocessed_im, *args, **kwargs)
  return network_fn


def extract_features(
        images,
        num_classes,
        output_stride=32,
        global_pool=True,
        model_variant=None,
        weight_decay=0.00001,
        dropout_keep_prob=0.5,
        regularize_depthwise=False,
        reuse=tf.AUTO_REUSE,
        is_training=False,
        fine_tune_batch_norm=False,
        cfg={}):
  """Extracts features by the particular model_variant."""
  if 'resnet' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay
    )
    features, end_points = get_network(
        model_variant, True, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=output_stride,
            reuse=reuse,
            scope=name_scope[model_variant])
  elif 'mobilenet_v2' in model_variant:
      arg_scope = arg_scopes_map[model_variant](
          is_training=(is_training and fine_tune_batch_norm),
          weight_decay=weight_decay,
          stddev=0.09,
          dropout_keep_prob=dropout_keep_prob,
          bn_decay=0.997
      )
      features, end_points = get_network(
          model_variant, True, arg_scope)(
          inputs=images,
          num_classes=num_classes,
          depth_multiplier=float(cfg['depth_multiplier']),
          scope=name_scope[model_variant],
          conv_defs=mobilenet_v2.V2_DEF,
          finegrain_classification_mode=False,
          min_depth=None,
          divisible_by=None,
          reuse=reuse,
          output_stride=output_stride,
          is_training=(is_training and fine_tune_batch_norm))
  else:
    raise ValueError('Unknown model variant %s.' % model_variant)

  return features, end_points



