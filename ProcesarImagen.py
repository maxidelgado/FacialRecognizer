import cv2
import sys
import numpy as np
import Clasificador as cl
import os

#Clasificadores
faceCascade = cv2.CascadeClassifier('Clasificadores/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('Clasificadores/haarcascade_eye_tree_eyeglasses.xml')

#Variables para calcular posicion de ojos
EYE_SX = 0.16
EYE_SY = 0.26
EYE_SW = 0.30
EYE_SH = 0.28

#Factor de escalador de imagen
reduccion = 480

DESIRED_LEFT_EYE_X = 0.19;
DESIRED_LEFT_EYE_Y = 0.19;
FaceWidth = 100
FaceHeight = 100

def DetectarCaras(label,frame):
    # Normalizar frame
    frame_norm = NormalizarImagen(frame)

    faces = faceCascade.detectMultiScale(
        frame_norm,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(30, 30),
        maxSize=(400, 400),
        flags=0
    )
    # Dibujar un rectangulo alrededor de la cara
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        wm = x + (w / 2)
        hm = y + (h / 2)
        frame_cara = DetectarOjos(cv2.getRectSubPix(frame_norm, (w, h), (wm, hm)))
        if np.size(frame_cara)>1:
            return frame_cara

def NormalizarImagen(frame):
    col = np.size(frame,1)
    row = np.size(frame, 0)
    if col > reduccion:
        # Reduce la escala a 480
        escala = col / reduccion
        escalarAlto = row / escala
        frame = cv2.resize(frame, (reduccion, escalarAlto))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    return gray_eq

def DetectarOjos(frame_cara):
    col = np.size(frame_cara, 1)
    row = np.size(frame_cara, 0)

    leftX = int(np.round(col * EYE_SX))
    topY = int(np.round(row * EYE_SY))
    widthX = int(np.round(col * EYE_SW))
    heightY = int(np.round(row * EYE_SH))
    rightX = int(np.round(col * (1 - EYE_SX - EYE_SW)))

    topLeftOfFace = cv2.getRectSubPix(frame_cara, (widthX, heightY), ((leftX + (widthX/2)), (topY + (heightY/2))))
    topRightOfFace = cv2.getRectSubPix(frame_cara, (widthX, heightY), ((rightX + (widthX/2)), (topY + (heightY/2))))

    eyeLeft = eye_cascade.detectMultiScale(
        topLeftOfFace
    )

    for (x,y,w,h) in eyeLeft:
        x = leftX + x
        y = topY + y
        #cv2.rectangle(frame_cara, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyeLeft[0][0] = x
        eyeLeft[0][1] = y

    eyeRight = eye_cascade.detectMultiScale(
        topRightOfFace
    )

    for (x, y, w, h) in eyeRight:
        x = rightX + x
        y = topY + y
        #cv2.rectangle(frame_cara, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyeRight[0][0] = x
        eyeRight[0][1] = y

    if np.size(eyeLeft) == 4 & np.size(eyeRight) == 4:
        return EstabilizarCara(frame_cara,eyeLeft,eyeRight)
    else:
         return None


def EstabilizarCara(frame_cara, leftEye, rightEye):
    #Se accede a las coordenadas, ancho y alto
    left = (leftEye[0][0] + leftEye[0][2] / 2, leftEye[0][1] + leftEye[0][3] / 2)
    right= (rightEye[0][0] + rightEye[0][2] / 2, rightEye[0][1] + rightEye[0][3] / 2)
    eyesCenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)

    #Calcular el angulo entre los dos ojos
    dy = (right[1] - left[1])
    dx = (right[0] - left[0])
    len = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X)

    desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * FaceWidth
    scale = desiredLen / len

    rot_mat = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    rot_mat[0][2] += FaceWidth * 0.5 - eyesCenter[0]
    rot_mat[1][2] += FaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter[1]
    face = cv2.warpAffine(frame_cara,rot_mat,(FaceWidth,FaceHeight))
    return face

def CargarCaras(path):
    images, labels = [],[]
    c = 0
    for directorio, directorios, nombreArchivos in os.walk(path):
        for subdirectorios in directorios:
            pathPersona = os.path.join(directorio,subdirectorios)
            for nombreArchivo in os.listdir(pathPersona):
                img = cv2.imread(os.path.join(pathPersona, nombreArchivo), cv2.IMREAD_GRAYSCALE)
                images.append(np.asarray(img, dtype=np.uint8))
                labels.append(c)
            c+=1
        return images, labels

def MarcarCara(frame, label):
    frame_norm = NormalizarImagen(frame)

    face = faceCascade.detectMultiScale(frame_norm, 1.2, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame_norm, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img = frame_norm,
                    text = str(label),
                    org = (x,y),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = 1,
                    color = (255,0,0))

    return frame_norm
