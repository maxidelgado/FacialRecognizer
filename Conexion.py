# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 14:03:03 2016

@author: gastr
"""

import MySQLdb as dbapi
import sys
import numpy as np
import cv2
import ProcesarImagen as pi
    

DB_HOST = 'localhost'
DB_USER = 'root' 
DB_PASS = '' 
DB_NAME = 'baseriv' 
 
#Ejecuta una consulta SQL que se le pasa por parametro y devuelve una tupla
def ejecutarConsulta(query=''): 
    datos = [DB_HOST, DB_USER, DB_PASS, DB_NAME] 
 
    conn = dbapi.connect(*datos) # Conectar a la base de datos 
    cursor = conn.cursor()         # Crear un cursor 
    cursor.execute(query)          # Ejecutar una consulta 
 
    if query.upper().startswith('SELECT'): 
        data = cursor.fetchall()   # Traer los resultados de un select 
    else: 
        conn.commit()              # Hacer efectiva la escritura de datos 
        data = None 
 
    cursor.close()                 # Cerrar el cursor 
    conn.close()                   # Cerrar la conexión 
 
    return data

#Retorna las imagenes y su etiqueta para entrenar un modelo en la BD
def getImagenesEntrenar():
    #Ejecuta la consulta que trae de la BD la imagen y su legajo
    resultado = ejecutarConsulta('SELECT legajo,imagen FROM carasclasificador')
    #Archivo donde se va a guardar la imagen
    nombreImg = 'imagen.jpg'     
    imgRes = open(nombreImg,'wb');
    
    listaCaras, labels = [],[]
    #Recorro las imagenes obtenidas
    for tupla in resultado:
        #Escribo en la imagen
        imgRes.write(tupla[1])
        #Leo la imagen en formato OpenCV
        img = cv2.imread(nombreImg)
        imgEst = pi.DetectarOjos(img)
        if(np.size(imgEst)>1):        
            listaCaras.append()
            labels.append(tupla[0])
    
    #Cierra el archivo       
    imgRes.close()
    return listaCaras,labels

#Guarda una cara detectada en la BD
def setImagenBD(legajo, imagen):
    consulta = "INSERT INTO carasdetectadas(legajo,imagen,fecha) VALUES(%i,'%s',NOW())" %(int(legajo),imagen)
    ejecutarConsulta(consulta)

#Retorna un diccionario que contiene los legajos de todas las personas
def getLegajos():
    resultado = ejecutarConsulta("SELECT legajo,nombre,apellido FROM personas")
    legajos = {}    
    for tupla in resultado:
        legajos.update(tupla[0],tupla[1] + ' ' + tupla[2])
    return legajos