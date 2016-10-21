import cv2
import sys
import numpy as np
import Capturador as cap
import ProcesarImagen as pi

listaID, listaCaras = [],[]
MODEL_FILE = "model.mdl"

def Entrenar(labels, listaCaras):
    model = cv2.face.createFisherFaceRecognizer()
    # model = cv2.createEigenFaceRecognizer()
    model.train(listaCaras, np.asarray(labels))
    model.save(MODEL_FILE)
    print 'Entrenado'
    return model

def Predecir(frame_cara):
    model = cv2.face.createFisherFaceRecognizer()
    model.load(MODEL_FILE)
    prediccion = -1
    prediccion = model.predict(frame_cara)
    return prediccion







