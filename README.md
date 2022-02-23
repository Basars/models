# Models

ML model weights and trainable codes for Basars

### Prepare Dependencies
```
pip install tensorflow requests pandas opencv-python
pip install git+https://github.com/Basars/trans-unet.git
pip install git+https://github.com/Basars/basars-addons.git
```

### Training Your Own Basars

Make sure you've installed `Python >= 3.8`.
```
usage: python -m basars.train 
                [-h] --type {stairs,proj}
                [--num-classes NUM_CLASSES] [--epochs EPOCHS] 
                [--batch_size BATCH_SIZE] [--buffer_size BUFFER_SIZE]
                [--multiprocessing-workers MULTIPROCESSING_WORKERS] [--cache-dataset CACHE_DATASET]

Polyp Segmentation and Phase Classification from Endoscopic Images

optional arguments:
  -h, --help            show this help message and exit
  --type {stairs,proj}  The type of transformer model. Default value is 'stairs'
  --num-classes NUM_CLASSES
                        Number of classes to be classified. Default value is 5
  --epochs EPOCHS       Epochs that how many times the model would be trained. Default value is 1290
  --batch_size BATCH_SIZE
                        The batch size. Default value is 64
  --buffer_size BUFFER_SIZE
                        The buffer size for shuffling datasets. Default value is 1024
  --multiprocessing-workers MULTIPROCESSING_WORKERS
                        Number of workers for prefetching datasets. Default value is 64
  --cache-dataset CACHE_DATASET
                        True to cache datasets on memory otherwise don't. Default value is True
```
#### Training Sample
```
python -m basars.train --type proj --epochs 1290
```


### Configuration Guide
Refer the repository: [final-experiments](https://github.com/Basars/final-experiments)

### Weights

- [basars-cls5-stairs.h5](https://github.com/Basars/models/releases/download/v1.0/basars-cls5-stairs.h5)
- [basars-cls5-proj.h5](https://github.com/Basars/models/releases/download/v1.0/basars-cls5-proj.h5)

You can find out the weights in [Releases](https://github.com/Basars/models/releases).

### Naming Convention and Meaning

`stairs` model have `conv3x3 (256, 128, 64, 32, 16) → conv1x1 (5)` upsamples

`proj` model have `conv3x3 (256, 128, 64, 16) → conv1x1 (5, 5)` upsamples

### Model Definition

`stairs` model:
```python
model = Sequential(name='ViT-stairs', layers=[
    VisionTransformer(input_shape=(224, 224, 3),
                      upsample_channels=(256, 128, 64, 32),
                      output_kernel_size=3, num_classes=16),
    Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
])
model.load_weights('basars-cls5-stairs.h5')
```

`proj` model:
```python
Sequential(name='ViT-proj', layers=[
    VisionTransformer(input_shape=(224, 224, 3), num_classes=5),
    Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
])
model.load_weights('basars-cls5-proj.h5')
```

### Model Architecture
![architecture](https://github.com/Basars/models/blob/main/static/architecture.png)

### Samples

<p align='center'>
  <img alt='sample0' src="https://github.com/Basars/models/blob/main/static/r0.png">
  <img alt='sample1' src="https://github.com/Basars/models/blob/main/static/r1.png">
  <img alt='sample2' src="https://github.com/Basars/models/blob/main/static/r2.png">
  <img alt='sample3' src="https://github.com/Basars/models/blob/main/static/r3.png">
</p>