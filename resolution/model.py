import keras


def get_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(81*9))
    model.add(keras.layers.Reshape((-1, 9)))
    model.add(keras.layers.Activation('softmax'))
    
    return model


 