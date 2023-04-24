import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Input
from tensorflow.keras.models import Model

def generator(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    num_channels = input_shape[2]
    x = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", activation="tanh")(x)

    model = Model(inputs=input, outputs=x)
    return model

def discriminator(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model