import os
import cv2

import pandas as pd
import tensorflow as tf
import numpy as np


def fetch_images_recursively(basedir):
    queries = []
    for dirname in os.listdir(basedir):
        pathname = os.path.join(basedir, dirname)
        if os.path.isdir(pathname):
            queries += fetch_images_recursively(pathname)
        else:
            queries.append(pathname)
    return queries


def prepare_datasets_on_memory(img_path='train', mask_path='train_masks', label_path='train_labels.csv'):
    img_paths = sorted(fetch_images_recursively(img_path))
    mask_paths = sorted(fetch_images_recursively(mask_path))
    df = pd.read_csv(label_path)
    return img_paths, mask_paths, df


def parse_image(image_path, mask_path, df, imgsize=224, num_classes=5):
    filename = image_path.split('/')[-1].split('.')[0]
    phase = df[df['filename'] == filename]['phase'].values[0]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(imgsize, imgsize), interpolation=cv2.INTER_AREA)
    # (224, 224, 3)

    dst = np.zeros((imgsize, imgsize, num_classes), dtype=np.uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, dsize=(imgsize, imgsize), interpolation=cv2.INTER_AREA)
    mask = np.reshape(mask, (imgsize, imgsize, 1))

    if phase > 0:
        dst[:, :, phase:phase + 1] = mask

    mask = 255 - mask  # all areas except cancer
    dst[:, :, 0:1] = mask
    # (224, 224, num_classes)
    return image, dst


def cache_datasets(img_paths, mask_paths, df, imgsize=224, num_classes=5):
    images = []
    for image_path, mask_path in zip(img_paths, mask_paths):
        images.append(parse_image(image_path, mask_path, df, imgsize, num_classes))
    return images


def dataset_generator(cache, img_paths, mask_paths, df, imgsize=224, num_classes=5):
    if cache:
        cached_images = cache_datasets(img_paths, mask_paths, df, imgsize, num_classes)

        def _generate():
            for image, mask in cached_images:
                yield image, mask
    else:
        def _generate():
            for image_path, mask_path in zip(img_paths, mask_paths):
                return parse_image(image_path, mask_path, df, imgsize, num_classes)

    return _generate


def parse_dataset(img_paths, mask_paths, df, cache=True, augment=False, imgsize=224, num_classes=5):
    dataset = tf.data.Dataset.from_generator(dataset_generator(cache, img_paths, mask_paths, df, imgsize, num_classes),
                                             output_signature=(tf.TensorSpec(shape=(imgsize, imgsize, 3),
                                                                             dtype=tf.uint8),
                                                               tf.TensorSpec(shape=(imgsize, imgsize, num_classes),
                                                                             dtype=tf.uint8)))

    @tf.function
    def _normalize(image, mask):
        image = tf.cast(image, tf.float32) / 255.
        mask = tf.cast(mask, tf.float32) / 255.

        return image, mask

    @tf.function
    def _augment(image, mask):
        image, mask = _normalize(image, mask)

        seed = tf.random.experimental.stateless_split(tf.random.uniform((2,), maxval=1000, dtype=tf.int32), num=1)[0, :]

        if tf.random.uniform(()) > 0.2:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform(()) > 0.2:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

        if tf.random.uniform(()) > 0.2:
            image = tf.image.stateless_random_crop(image, size=tf.shape(image), seed=seed)
            mask = tf.image.stateless_random_crop(mask, size=tf.shape(mask), seed=seed)

        if tf.random.uniform(()) > 0.2:
            image = tf.image.random_brightness(image, 0.2)

        if tf.random.uniform(()) > 0.2:
            k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k)
            mask = tf.image.rot90(mask, k)

        if tf.random.uniform(()) > 0.5:
            noise = tf.random.normal(tf.shape(image), mean=0., stddev=0.05, dtype=tf.float32)
            image = image + noise
            image = tf.clip_by_value(image, 0., 1.)

        return image, mask

    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(_normalize)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    return dataset.with_options(options)


def load_dataset(train_paths, test_paths, valid_paths, cache=True, imgsize=224, num_classes=5):
    img_paths, mask_paths, df = prepare_datasets_on_memory(*train_paths)
    train_dataset = parse_dataset(img_paths, mask_paths, df, cache, augment=True, imgsize=imgsize, num_classes=num_classes)

    datasets = {'train': train_dataset}
    for key, paths in zip(['test', 'valid'], [test_paths, valid_paths]):
        img_paths, mask_paths, df = prepare_datasets_on_memory(*paths)
        dataset = parse_dataset(img_paths, mask_paths, df, cache, imgsize=imgsize, num_classes=num_classes)
        datasets[key] = dataset
    return datasets
