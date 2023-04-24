from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Activation
from keras.models import Model

def generator(input_shape, latent_dim):
    input = Input(shape=(latent_dim,))
    
    x = Dense(256 * input_shape[0] // 4 * input_shape[1] // 4)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((input_shape[0] // 4, input_shape[1] // 4, 256))(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(input_shape[2], kernel_size=3, strides=1, padding="same")(x)
    x = Activation("tanh")(x)

    model = Model(inputs=input, outputs=x)
    return model

def discriminator(input_shape):
    input = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input, outputs=x)
    return model