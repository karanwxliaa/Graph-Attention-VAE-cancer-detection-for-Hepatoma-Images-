import tensorflow as tf
import numpy as np



# Preprocess the images: rescale pixel values (currently being used)
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Rescale to [0, 1]
    return image, label


# Preprocess and augment the images (not used now, just adding in case we decide to augment images for some reason later)
def preprocess_and_augment_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Rescale to [0, 1]
    
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random vertical flip
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Random hue
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    
    # Random crop
    crop_size = tf.random.uniform(shape=[], minval=int(0.8 * 1600), maxval=1600, dtype=tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, [1600, 1600])
    
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random translation
    translations = tf.random.uniform(shape=[2], minval=-50, maxval=50, dtype=tf.int32)
    image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.convert_to_tensor([1, 0, translations[0], 0, 1, translations[1]], dtype=tf.float32),
        output_shape=tf.convert_to_tensor([1600, 1600], dtype=tf.int32)
    )[0]
    
    # Random zoom
    scales = list(np.arange(0.8, 1.2, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[i] = [y1, x1, y2, x2]
    def random_zoom(image):
        def apply_zoom(image, boxes):
            choice = tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
            box = boxes[choice]
            return tf.image.crop_and_resize([image], [box], [0], [1600, 1600])[0]
        image = apply_zoom(image, boxes)
        return image
    image = random_zoom(image)
    
    return image, label
