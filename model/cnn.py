from keras.src.initializers.initializers import TruncatedNormal
from keras.src.layers.core import Activation, Flatten, Dropout, Dense
from keras import Sequential
from keras import regularizers
from keras.src.layers import Conv2D, MaxPooling2D, BatchNormalization


class SimpleCNNNet:

    @staticmethod
    def build(classes, kpi):
        width, height, depth = kpi["width"], kpi["height"], kpi["depth"]
        ki = TruncatedNormal(mean=0.0, stddev=kpi["stddev"])
        kr = regularizers.l2(kpi["l2"])

        model = Sequential()
        input_shape = (height, width, depth)
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=ki))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=ki, kernel_regularizer=kr))
        model.add(Activation("relu"))
        model.add(Dropout(kpi["dropout"]))

        model.add(Dense(classes, kernel_initializer=ki, kernel_regularizer=kr))
        model.add(Activation("softmax"))

        return model
