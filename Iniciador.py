import cv2
import sys
import numpy as np
import Capturador as cap
import Clasificador as cl
import ProcesarImagen as proc

listaCaras, labels = [],[]

opcion = raw_input('Entrenar (e)  | Detectar (d):\n')

while True:
    if opcion == 'e':
        agregar = raw_input('Agregar persona?(s/n):\n')
        if agregar == 's':
            nombre = raw_input('Nombre de la persona:')
            cantidad = raw_input('Cantidad de imagenes:')
            cap.ModoEntrenamiento(str(nombre),int(cantidad))
            opcion = raw_input('Entrenar (e)  | Detectar (d):\n')
        else:
            listaCaras, labels = proc.CargarCaras('Caras/')
            cl.Entrenar(labels,listaCaras)
            opcion = raw_input('Entrenar (e)  | Detectar (d):\n')
    elif opcion == 'd':
        cap.CapturarVideo(True)
    else:
        print 'Opcion no valida...'
        opcion = raw_input('Entrenar (e)  | Detectar (d):\n')
