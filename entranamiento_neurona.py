from turtle import color
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns

class Clasificador:
   
    def prediccion_con_train(self):
        tipo_de_datos_train  = []
        imagenes_train   = []
        categorias_train = os.listdir('./dataset/train') 
        print(categorias_train)

        contador = 0
        for path in categorias_train :
            for imagen in os.listdir('./dataset/train/'+path) :
                img = Image.open('./dataset/train/'+"/"+imagen).resize((100,100)) 
                img = np.asarray(img)
                if len(img.shape) == 3:
                    img = img[:,:,0]
                imagenes_train.append(img)
                tipo_de_datos_train.append(contador)
            contador += 1
        print(tipo_de_datos_train)

        imagenes_train = np.asanyarray(imagenes_train)
        imagenes_train.shape

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,(5,5),input_shape=(100,100,4)),
            tf.keras.layers.MaxPooling2D(3,3),
            tf.keras.layers.Conv2D(64,(5,5)),
            tf.keras.layers.MaxPooling2D(3,3),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Flatten(input_shape=(28,28)), 
            tf.keras.layers.Dense(100, activation='relu'), 
            # tf.keras.layers.Dense(128, activation='relu'), 
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'), 
        ])

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        tipo_de_datos_train = np.asarray(tipo_de_datos_train)
        print(tipo_de_datos_train)

        historial = model.fit(imagenes_train, tipo_de_datos_train, epochs=10) 
        plt.plot(historial.history['loss'], color='pink')
        plt.show()
        return model


    