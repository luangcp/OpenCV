""" UTILIZANDO A WEBCAM PARA RASTREAMENTO"""
""" CAMSHIFT É UMA MELHORIA DO MEANSHIFT"""
"""UTILIZA A CAMERA DA WEB CAM PARA RASTREAR """
# OBS: Instalar o imutils pip install imutils

import numpy as np
import cv2
import time
from imutils.video import VideoStream  # Recursos de video


# INICIALIZANDO A CAMERA
cap = VideoStream(src=0).start()  # Iniciando a webcam variavel de captura
time.sleep(1.0)  # configurando um tempo pra webcam se adaptar abrir direitinho

cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # ret(return) retorna True ou False

bbox = cv2.selectROI(frame, False)  # o false é pra caixa delimitadora n ter a cruz no meio
x, y, w, h = bbox  # xywh são as variaveis de um bbox, eixo x, eixo y, altura e largura
track_window = (x, y, w, h)
# print(track_window)

roi = frame[y:y+h, x:x+w]  # selecionando a região de interesse da imagem
# y+h é a altura . x+w a largura
# cv2.imshow('ROI', roi)
# cv2.waitKey(0)

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# LEMBRANDO QUE AS CORES NO openCV é ao contrario BGR
# cv2.imshow("ROI HSV", hsv_roi)
# cv2.waitKey(0)

# GERANDO O HISTOGRAMA
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180]) # histograma hsv, num de canais, mascara, densidade, range


# baixar a biblioteca matplotlib -> pip install matplotlib
# ver o histograma com o matplotlib

# vizualizando
# vai criar um histograma com as cores
import matplotlib.pyplot as plt
plt.hist(roi.ravel(), 180, [0, 180])
# plt.show()
# cv2.waitKey(0)

# Normalizando o histograma para valores entre 0 e 1
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# criterio de parada -> quando vc quer q o algoritmo pare
# primeiro parametro 10
# segundo parametro 1
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# TERM_CRITERIA_EPS -> Quantidade de repetições
# TERM_CRITERIA_COUNT -> MAIS SENSIVEL AS MUDANÇAS NOS PIXELS

while True:
    ret, frame = cap.read()  # em ret retorna true ou false. se for true conseguiu a captura
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # calculo da densidade:
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 100], 1)  # dst é densidade
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('Camshift Rastreador', img2)

        if cv2.waitKey(1) == 13:  # tecla enter
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()


