import os
import requests

from urllib import request
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from transunet import VisionTransformer


PRETRAINED_MODEL_DIR = 'models'


def download_file(filename, url, error):
    print('Downloading {} to {}'.format(url, filename))
    try:
        request.urlretrieve(url, filename)
    except:
        os.unlink(filename)
        print('ERROR: {}'.format(error))


def attempt_to_download(filename, repo='Basars/models'):
    if not os.path.exists(PRETRAINED_MODEL_DIR):
        os.mkdir(PRETRAINED_MODEL_DIR)
    filepath = os.path.join(PRETRAINED_MODEL_DIR, filename)
    if os.path.exists(filepath):
        return filepath

    response = requests.get('https://api.github.com/repos/{}/releases/latest'.format(repo)).json()
    assets = [x['name'] for x in response['assets']]
    tag = response['tag_name']

    if filename in assets:
        download_file(filepath,
                      url='https://github.com/{}/releases/download/{}/{}'.format(repo, tag, filename),
                      error='Failed to download {}, try downloading from https://github.com/{}/releases'
                      .format(filename, repo))
        return filepath
    return None


def create_proj_vision_transformer(name='basars-proj'):
    model = Sequential(name=name, layers=[
        VisionTransformer(input_shape=(224, 224, 3),
                          num_classes=5,
                          encoder_trainable=False),
        Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
    ])
    filepath = attempt_to_download('basars-cls5-proj.h5')
    model.load_weights(filepath)
    model.trainable = False
    return model


def create_stairs_vision_transformer(name='basars-stairs'):
    model = Sequential(name=name, layers=[
        VisionTransformer(input_shape=(224, 224, 3),
                          upsample_channels=(256, 128, 64, 32),
                          output_kernel_size=3,
                          num_classes=16,
                          encoder_trainable=False),
        Conv2D(5, kernel_size=(1, 1), padding='same', activation='softmax', use_bias=False)
    ])
    filepath = attempt_to_download('basars-cls5-stairs.h5')
    model.load_weights(filepath)
    model.trainable = False
    return model
