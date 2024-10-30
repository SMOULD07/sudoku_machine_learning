import tensorflow as tf
from tensorflow import keras
import keras
import numpy as np 
from model import get_model 
from processing import get_data 
from outils import *

#chargement des données 
x_train, x_test, y_train, y_test = get_data("data/sudoku.csv")


# entrainement du modele 
model= get_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001))
model.fit(x_train, y_train, batch_size=32, epochs=2)



model.save('sudoku_model.h5') 
print("modele mis a jour")

