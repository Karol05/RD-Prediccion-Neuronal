from turtle import color
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns

from entranamiento_neurona import Clasificador

class NeuronaEntrenada:

    def neurona_entrenada(self,model):
        tipo_de_datos_test = []
        imagenes_test = []

        categorias_test = os.listdir('dataset/test') 
        contador_test = 0
        for path in categorias_test :
            for imagen in os.listdir('dataset/test/'+path) :
                img = Image.open('dataset/test/'+path+"/"+imagen).resize((100,100))
                img = np.asarray(img)
                if len(img.shape) == 3:
                    img = img[:,:,0]
                imagenes_test.append(img)
                tipo_de_datos_test.append(contador_test)
            contador_test += 1
            
        imagenes_test = np.array(imagenes_test)
        tipo_de_datos_test = np.array(tipo_de_datos_test)
        y_predict = np.argmax(model.predict(imagenes_test), axis=1)
        y_really = tipo_de_datos_test

        print(y_predict)  
        print(y_really) 
       
        mis_categorias = 'Limon'
        matriz = tf.math.confusion_matrix(y_really, y_predict)
        figura_matriz = plt.figure(figsize=(6,6))
        sns.heatmap(matriz, xticklabels=mis_categorias, yticklabels=mis_categorias, annot=True, fmt="g", cmap="crest")
        plt.xlabel('Prediccion')
        plt.ylabel('Label')
        plt.show()
     
        test = ['.\\dataset\\test\\limon_01.png','.\\dataset\\test\\limon_02.png','.\\dataset\\test\\hoja_03.png','.\\dataset\\test\\limon_11.png','.\\dataset\\limon_30.png']
        for i in test :
            img = Image.open(i).resize((28,28))
            img = np.asarray(img)
            if len(img.shape) == 3:
                img = img[:,:,0]
            img = np.array([img]) 
            predicciones = model.predict(img)
            print(categorias_test[np.argmax(predicciones[0])])
        

if __name__ == '__main__':   
    neurona = Clasificador()
    modelo = neurona.prediccion_con_train()
    entranada = NeuronaEntrenada()
    entranada.neurona_entrenada(modelo)

