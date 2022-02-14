# Models

ML model weights for Basars

## Weights

- [basars-cls5-stairs.h5](https://github.com/Basars/models/releases/download/v1.0/basars-cls5-stairs.h5)
- [basars-cls5-proj.h5](https://github.com/Basars/models/releases/download/v1.0/basars-cls5-proj.h5)

You can find out the weights in [Releases](https://github.com/Basars/models/releases).

## Naming Convention

`stairs` model have `conv3x3 (256, 128, 64, 32, 16) → conv1x1 (5)` upsamples

`proj` model have `conv3x3 (256, 128, 64, 16) → conv1x1 (5, 5)` upsamples

## Model Definition

`stairs` model:
```python
model = Sequential(name='ViT-stairs', layers=[
    VisionTransformer(input_shape=(224, 224, 3), num_classes=5),
    Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
])
model.load_weights('basars-cls5-stairs.h5')
```

`proj` model:
```python
Sequential(name='ViT-proj', layers=[
    VisionTransformer(input_shape=(224, 224, 3), 
                      upsample_channels=(256, 128, 64, 32), 
                      output_kernel_size=3, num_classes=16),
    Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
])
model.load_weights('basars-cls5-proj.h5')
```

## Samples

![r0](https://github.com/Basars/models/blob/main/static/r0.png)
![r1](https://github.com/Basars/models/blob/main/static/r1.png)
![r2](https://github.com/Basars/models/blob/main/static/r2.png)
![r3](https://github.com/Basars/models/blob/main/static/r3.png)