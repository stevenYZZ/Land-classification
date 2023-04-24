import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def cnn2d(input_shape, num_class, w_decay=0):
    model = tf.keras.Sequential()

    # 第一层卷积
    model.add(Conv2D(filters=50, kernel_size=(5, 5), input_shape=input_shape))
    model.add(Activation('relu'))

    # 第二层卷积
    model.add(Conv2D(filters=100, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten层
    model.add(Flatten())

    # 全连接层
    model.add(Dense(units=100, kernel_regularizer=regularizers.l2(w_decay)))
    model.add(Activation('relu'))
    model.add(Dense(units=num_class, activation='softmax'))

    # 编译模型
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    return model

# def cnn2d(input_shape, num_classes):
#     model = Sequential()
#     model.add(InputLayer(input_shape=input_shape))
#     model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(Conv2D(filters=100, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(units=num_classes, activation='softmax'))

#     return model