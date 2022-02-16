import argparse

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from basars_addons.losses import Dice
from basars_addons.schedules import CosineDecayWarmupRestarts
from basars.models import create_stairs_vision_transformer, create_proj_vision_transformer
from basars.metrics import MaskedThresholdBinaryIoU
from basars.datasets import load_dataset


parser = argparse.ArgumentParser(description='Polyp Segmentation and Phase Classification from Endoscopic Images',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--type',
                    required=True,
                    choices=['stairs', 'proj'],
                    default='stairs',
                    help='The type of transformer model')
parser.add_argument('--num-classes',
                    default=5,
                    help='Number of classes to be classified.')
parser.add_argument('--epochs',
                    default=1290,
                    help='Epochs that how many times the model would be trained')
parser.add_argument('--batch_size',
                    default=64,
                    help='The batch size')
parser.add_argument('--buffer_size',
                    default=1024,
                    help='The buffer size for shuffling datasets')
parser.add_argument('--multiprocessing-workers',
                    default=64,
                    help='Number of workers for prefetching datasets')
parser.add_argument('--cache-dataset',
                    default=True,
                    help='True to cache datasets on memory otherwise don\'t')

args = parser.parse_args()

model_type = args.type
num_classes = int(args.num_classes)
model_name = 'basars-{}'.format(model_type)
buffer_size = int(args.buffer_size)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
multiprocessing_workers = int(args.multiprocessing_workers)
cache = args.cache_dataset
imgsize = 224

print()
print('Arguments Information:')
print('Model name: {}'.format(model_name))
print('Number of classes: {}'.format(num_classes))
print('Batch size: {}'.format(batch_size))
print('Buffer size: {}'.format(buffer_size))
print('Epochs: {}'.format(epochs))
print('Number of multiprocessing workers: {}'.format(multiprocessing_workers))

# TODO: Change to command-line arguments
train_paths = ('datasets/train', 'datasets/train_masks', 'datasets/train_labels.csv')
test_paths = ('datasets/test', 'datasets/test_masks', 'datasets/test_labels.csv')
valid_paths = ('datasets/valid', 'datasets/valid_masks', 'datasets/valid_labels.csv')

if model_type == 'stairs':
    model_loader = create_stairs_vision_transformer
elif model_type == 'proj':
    model_loader = create_proj_vision_transformer
else:
    raise ValueError('Invalid model type: {}. Only \'stairs\' and \'proj\' are valid.'.format(model_type))

print()
print('Preparing strategy for distributed training if available...')
strategy = tf.distribute.MirroredStrategy()
batch_size *= strategy.num_replicas_in_sync

print()
print('Compiling \'{}\'...'.format(model_name))
scheduler = CosineDecayWarmupRestarts(100, initial_learning_rate=1e-3, first_decay_steps=300, t_mul=1.0)

with strategy.scope():
    cce = CategoricalCrossentropy(reduction=Reduction.NONE)
    dice = Dice(num_classes=num_classes, reduction=Reduction.NONE)

    @tf.function
    def loss_fn(y_true, y_pred):
        cce_loss = cce(y_true, y_pred)
        dice_loss = dice(y_true, y_pred)

        cce_loss = tf.reduce_mean(cce_loss)
        dice_loss = dice_loss

        return cce_loss * 0.5 + dice_loss * 0.5

    optimizer = SGD(learning_rate=scheduler, momentum=0.9)

    model = model_loader(name=model_name)
    model.compile(optimizer, loss_fn,
                  metrics=[MaskedThresholdBinaryIoU(num_classes=num_classes - 1,
                                                    mask=slice(1, num_classes),
                                                    name='binary_iou')])
print('Compiled \'{}\''.format(model_name))

callbacks = [ModelCheckpoint('{}-best-loss.h5'.format(model_name),
                             monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1, save_weights_only=True),
             ModelCheckpoint('{}-best-iou.h5'.format(model_name),
                             monitor='val_binary_iou', mode='max',
                             save_best_only=True, verbose=1, save_weights_only=True),
             TensorBoard('./logs')]

print()
print('Preparing datasets...')

datasets, num_images, num_masks = load_dataset(train_paths, test_paths, valid_paths,
                                               cache=cache, imgsize=imgsize, num_classes=num_classes)
print('{} datasets prepared.'.format(len(datasets)))

train_dataset = datasets['train']
valid_dataset = datasets['valid']
test_dataset = datasets['test']

steps_per_epoch = int(num_images / batch_size)

train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

print()
print('Training model: {}'.format(model_name))
model.fit(train_dataset,
          epochs=epochs,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks,
          validation_data=valid_dataset,
          use_multiprocessing=True,
          workers=multiprocessing_workers)

print('Finished training.')

print()
print('Evaluating model...')
loss, binary_iou = model.evaluate(test_dataset)
print('Loss: {}'.format(loss))
print('IoU: {}'.format(binary_iou))

print()
print('Saving the latest checkpoint')
model.save('{}-latest.h5'.format(model_name))

print('All jobs have been finished')
