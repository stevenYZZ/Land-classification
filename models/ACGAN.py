from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Input, Reshape, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def acgan_generator(input_shape):
    noise = Input(shape=input_shape)
    num_classes = input_shape[-1]

    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(noise)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_channels = input_shape[2]
    x = Conv2DTranspose(num_channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    out = Activation('tanh')(x)

    model = Model(inputs=noise, outputs=out)
    return model


def acgan_discriminator(input_shape, num_classes):
    img = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    # Output layer for real/fake prediction
    validity = Dense(1, activation='sigmoid')(x)

    # Output layer for class prediction
    class_pred = Dense(num_classes, activation='softmax')(x)

    # Concatenate both outputs
    combined_outputs = Concatenate(axis=1)([validity, class_pred])

    model = Model(inputs=img, outputs=combined_outputs)
    return model