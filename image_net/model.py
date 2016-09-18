import os

import numpy
import requests
from PIL import Image
from keras import backend
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


def _vgg_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def _retrieve_model():
    path_to_model_file = os.path.join(os.path.dirname(__file__), 'vgg16_weights.h5')
    if not os.path.isfile(path_to_model_file):
        with open(path_to_model_file, 'wb') as handle:
            response = requests.get(
                'https://s3-eu-west-1.amazonaws.com/ai-cooking/image_net_model/vgg16_weights.h5',
                stream=True
            )

            if not response.ok:
                raise Exception("Couldn't download the VGG model")

            for block in response.iter_content(1024):
                handle.write(block)

    model = _vgg_16(path_to_model_file)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def _image_to_raw_features(image_object):
    new_width = 224
    new_height = 224
    image_tensor = image_object.resize((new_width, new_height), Image.ANTIALIAS)
    image_tensor = numpy.array(image_tensor)

    # The mean pixel values are taken from the VGG authors,
    # which are the values computed from the training data set.
    mean_pixel = [103.939, 116.779, 123.68]
    image_tensor = image_tensor.astype(numpy.float32, copy=False)
    for c in range(3):
        image_tensor[:, :, c] = image_tensor[:, :, c] - mean_pixel[c]

    image_tensor = image_tensor.transpose((2, 0, 1))

    return numpy.expand_dims(image_tensor, axis=0)


class ImageNetModel(object):
    def __init__(self):
        self.model = _retrieve_model()
        n_th_layer = 33
        self.get_features = backend.function(
            [self.model.layers[0].input, backend.learning_phase()], [self.model.layers[n_th_layer].output, ]
        )

    def extract_features(self, image_object):
        return self.get_features([_image_to_raw_features(image_object), 0])[0][0].astype(float)
