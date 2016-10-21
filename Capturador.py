import cv2
import sys
import numpy as np
import ProcesarImagen as pi
import Clasificador as cl
import os

def CapturarVideo(abrir):
    label=0
    #Capturar video de la webcam
    video_capture = cv2.VideoCapture(0) #0 es la webcam por defecto (integrada)
    while abrir:
        label+=1
        # Capturar frame por frame
        ret, frame = video_capture.read()

        frame_cara = pi.DetectarCaras(label,frame)

        if np.size(frame_cara) > 1:
            predict = cl.Predecir(frame_cara)
            frame_norm = pi.MarcarCara(frame,predict)
            cv2.imshow('Imagen normalizada',frame_norm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def ModoEntrenamiento(nombre,cantidadImagenes):
    labels,listaCaras = [],[]
    os.mkdir('Caras/'+str(nombre))
    # Capturar video de la webcam
    video_capture = cv2.VideoCapture(0)  # 0 es la webcam por defecto (integrada)
    label = 0
    while label < cantidadImagenes:
        # Capturar frame por frame
        ret, frame = video_capture.read()

        # Mostrar el frame actual
        # cv2.imshow('Video', frame)

        frame_cara = pi.DetectarCaras(label, frame)
        if np.size(frame_cara)>1:
            cv2.imshow('cara',frame_cara)
            cv2.imwrite('Caras/'+str(nombre)+'/'+str(label)+'.jpg', frame_cara)
            label+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()






