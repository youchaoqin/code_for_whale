"""some data tranformation fuctions"""

import tensorflow as tf

def _rotate_180(processed_image, prob=0.5):
    with tf.name_scope('rotate_180'):
        do_rotate = tf.less(tf.random_uniform([]), prob)
        processed_image = tf.cond(pred=do_rotate,
                                  true_fn=lambda: tf.image.rot90(processed_image, 2),
                                  false_fn=lambda: tf.identity(processed_image))
    return processed_image

def _random_bhsc(image):
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image

def _random_sbch(image):
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def _random_chbs(image):
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image

def _random_hscb(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return image

def _do_random_color_distort(processed_image):
    random_num = tf.random_uniform([])

    image = tf.identity(processed_image)

    image = tf.cond(pred=(tf.less_equal(random_num, 0.25)),
                    true_fn=lambda: _random_bhsc(image),
                    false_fn=lambda: image)
    image = tf.cond(pred=tf.logical_and(tf.greater(random_num, 0.25),
                                         tf.less_equal(random_num, 0.5)),
                    true_fn=lambda: _random_chbs(image),
                    false_fn=lambda: image)
    image = tf.cond(pred=tf.logical_and(tf.greater(random_num, 0.5),
                                         tf.less_equal(random_num, 0.75)),
                    true_fn=lambda: _random_hscb(image),
                    false_fn=lambda: image)
    image = tf.cond(pred=tf.logical_and(tf.greater(random_num, 0.75),
                                         tf.less_equal(random_num, 1.0)),
                    true_fn=lambda: _random_sbch(image),
                    false_fn=lambda: image)
    return image


def random_distort_color(processed_image, distort_prob=0.5):
    with tf.name_scope('random_distort_color'):
        random_value = tf.random_uniform([])
        do_distort = tf.less_equal(random_value, distort_prob)
        distorted = tf.cond(pred=do_distort,
                            true_fn= lambda: _do_random_color_distort(processed_image),
                            false_fn= lambda: processed_image)
    return distorted


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
  """Gets a random scale value.

  Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.

  Returns:
    A random scale value selected between minimum and maximum value.

  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
  if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
    raise ValueError('Unexpected value of min_scale_factor.')

  if min_scale_factor == max_scale_factor:
    return tf.to_float(min_scale_factor)

  # When step_size = 0, we sample the value uniformly from [min, max).
  if step_size == 0:
    return tf.random_uniform([1],
                             minval=min_scale_factor,
                             maxval=max_scale_factor)

  # When step_size != 0, we randomly select one discrete value from [min, max].
  num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
  scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
  shuffled_scale_factors = tf.random_shuffle(scale_factors)
  return shuffled_scale_factors[0]


def randomly_scale_image(image, scale=1.0):
  """Randomly scales image and label.

  Args:
    image: Image with shape [height, width, 3].
    scale: The value to scale image and label.

  Returns:
    Scaled image and label.
  """
  # No random scaling if scale == 1.
  if scale == 1.0:
    return image
  image_shape = tf.shape(image)
  new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

  # Need squeeze and expand_dims because image interpolation takes
  # 4D tensors as input.
  image = tf.squeeze(tf.image.resize_bilinear(
      tf.expand_dims(image, 0),
      new_dim,
      align_corners=True), [0])
  return image


def flip_dim(tensor, prob=0.5, dim=1):
  random_value = tf.random_uniform([])

  is_flipped = tf.less_equal(random_value, prob)
  flipped = tf.cond(is_flipped,
                    lambda: tf.reverse_v2(tensor, [dim]),
                    lambda: tensor)
  return flipped


def image_augmentation(image, aug_opt):
  """do image augmentation.

  Args:
    image: Input image.
    aug_opt: augmentation options a dictionary
  """
  with tf.name_scope('image_augmentation'):
      # Data augmentation by randomly scaling the inputs.
      random_scale_limits = aug_opt.get('random_scale', None)
      if random_scale_limits is not None:
          print('### apply random scale in {} ###'.format(random_scale_limits))
          scale = get_random_scale(random_scale_limits[0], random_scale_limits[1],
              aug_opt.get('random_scale_step_size', 0))
          image = randomly_scale_image(image, scale)

      # Data augmentation by h-flip or/and v-flip
      h_flip_prob = aug_opt.get('h_flip_prob', None)
      v_flip_prob = aug_opt.get('v_flip_prob', None)
      if h_flip_prob is not None:
          print('### apply random h-flip with prob=%f ###'%(h_flip_prob))
          image = flip_dim(image, prob=h_flip_prob, dim=1)
      if v_flip_prob is not None:
          print('### apply random v-flip with prob=%f ###'%(v_flip_prob))
          image = flip_dim(image, prob=v_flip_prob, dim=0)

      # Data augmentation by random rotate 180 degree
      rotate180_prob = aug_opt.get('rotate180_prob', None)
      if rotate180_prob is not None:
          print('### apply random rotate180 with prob=%f ###' % (rotate180_prob))
          image = _rotate_180(image,prob=rotate180_prob)

      # Data augmentation by random distrot color
      distort_color_prob = aug_opt.get('distort_color_prob', None)
      if distort_color_prob is not None:
          print('### apply random distort color with prob=%f ###' % (distort_color_prob))
          image = random_distort_color(image, distort_prob=distort_color_prob)

      image = tf.identity(image, name='AugmentedImage')

  return image


def resolve_shape(tensor, rank=None, scope=None):
  """Fully resolves the shape of a Tensor.

  Use as much as possible the shape components already known during graph
  creation and resolve the remaining ones during runtime.

  Args:
    tensor: Input tensor whose shape we query.
    rank: The rank of the tensor, provided that we know it.
    scope: Optional name scope.

  Returns:
    shape: The full shape of the tensor.
  """
  with tf.name_scope(scope, 'resolve_shape', [tensor]):
    if rank is not None:
      shape = tensor.get_shape().with_rank(rank).as_list()
    else:
      shape = tensor.get_shape().as_list()

    if None in shape:
      shape_dynamic = tf.shape(tensor)
      for i in range(len(shape)):
        if shape[i] is None:
          shape[i] = shape_dynamic[i]

    return shape


def resize_to_range(image, min_size, max_size):
    with tf.name_scope('resize_to_range', [image]):
        if (min_size is None) and (max_size is None):
            return image
        else:
            [orig_height, orig_width, _] = resolve_shape(image, rank=3)
            orig_height = tf.to_float(orig_height)
            orig_width = tf.to_float(orig_width)
            orig_min_size = tf.minimum(orig_height, orig_width)
            orig_max_size = tf.maximum(orig_height, orig_width)

            # different case
            if (min_size is not None) and (max_size is None):
                min_size = tf.to_float(min_size)
                large_scale_factor = min_size / orig_min_size
                large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
                large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
                large_size = tf.stack([large_height, large_width])
                resized_im = tf.image.resize_bilinear(tf.expand_dims(image,axis=0),
                                                      large_size, align_corners=True)
            elif (min_size is None) and (max_size is not None):
                max_size = tf.to_float(max_size)
                small_scale_factor = max_size / orig_max_size
                small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
                small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
                small_size = tf.stack([small_height, small_width])
                resized_im = tf.image.resize_bilinear(tf.expand_dims(image,axis=0),
                                                      small_size, align_corners=True)
            elif (min_size is not None) and (max_size is not None):
                #larger size
                min_size = tf.to_float(min_size)
                large_scale_factor = min_size / orig_min_size
                large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
                large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
                large_size = tf.stack([large_height, large_width])
                #small size
                max_size = tf.to_float(max_size)
                small_scale_factor = max_size / orig_max_size
                small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
                small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
                small_size = tf.stack([small_height, small_width])
                #do resize
                new_size = tf.cond(
                    tf.to_float(tf.reduce_max(large_size)) > max_size,
                    lambda: small_size, lambda: large_size)
                resized_im = tf.image.resize_bilinear(tf.expand_dims(image,axis=0),
                                                      new_size, align_corners=True)
            else:
                raise Exception('Un-known case, check min_size, max_size')

            return tf.squeeze(resized_im, [0])


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
  """Pads the given image with the given pad_value.

  Works like tf.image.pad_to_bounding_box, except it can pad the image
  with any given arbitrary pad value and also handle images whose sizes are not
  known during graph construction.

  Args:
    image: 3-D tensor with shape [height, width, channels]
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    pad_value: Value to pad the image tensor with.

  Returns:
    3-D tensor of shape [target_height, target_width, channels].

  Raises:
    ValueError: If the shape of image is incompatible with the offset_* or
    target_* arguments.
  """
  image_rank = tf.rank(image)
  image_rank_assert = tf.Assert(
      tf.equal(image_rank, 3),
      ['Wrong image tensor rank [Expected] [Actual]',
       3, image_rank])
  with tf.control_dependencies([image_rank_assert]):
    image -= pad_value
  image_shape = tf.shape(image)
  height, width = image_shape[0], image_shape[1]
  target_width_assert = tf.Assert(
      tf.greater_equal(
          target_width, width),
      ['target_width must be >= width'])
  target_height_assert = tf.Assert(
      tf.greater_equal(target_height, height),
      ['target_height must be >= height'])
  with tf.control_dependencies([target_width_assert]):
    after_padding_width = target_width - offset_width - width
  with tf.control_dependencies([target_height_assert]):
    after_padding_height = target_height - offset_height - height
  offset_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(after_padding_width, 0),
          tf.greater_equal(after_padding_height, 0)),
      ['target size not possible with the given target offsets'])

  height_params = tf.stack([offset_height, after_padding_height])
  width_params = tf.stack([offset_width, after_padding_width])
  channel_params = tf.stack([0, 0])
  with tf.control_dependencies([offset_assert]):
    paddings = tf.stack([height_params, width_params, channel_params])
  padded = tf.pad(image, paddings)
  return padded + pad_value


def corp_image(image, crop_size, random_crop=None):
    crop_height = crop_size[0]
    crop_width = crop_size[1]

    # pad the image_size not smaller than corp_size
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
    image = pad_to_bounding_box(  # Pad image with mean pixel value.
        image, 0, 0, target_height, target_width, mean_pixel)

    # do crop
    if random_crop:
        offset_height = tf.random_uniform(
            [], maxval=target_height-crop_height+1, dtype=tf.int32)
        offset_width = tf.random_uniform(
            [], maxval=target_width-crop_width + 1, dtype=tf.int32)
        cropped_image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, crop_height, crop_width)
    else:
        cropped_image = tf.image.crop_to_bounding_box(
            image, 0, 0, crop_height, crop_width)

    return cropped_image
