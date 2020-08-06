"""
ASSEMELHASSE AO SPARCE POR VIDEO MAS FUNCIONA COMO UM "MAPA DE CALOR" o fundo
fica preto e as pessoas viram pixels luminosos
É mais preciso
MUITO UTILIZADO PARA SISTEMAS DE SEGURANÇA
"""
import cv2
import numpy as np

cap = cv2.VideoCapture("videos/walking.avi")
ret, first_frame = cap.read()  # Buscando o primeiro frame
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # Convertendo para escala de cinza

hsv = np.zeros_like(first_frame)

hsv[..., 1] = 255  # SATURAÇÃO


while True:
    ret, frame = cap.read()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame_gray_init,
                                        frame_grey,
                                        None,
                                        0.5,
                                        3,
                                        15,
                                        3,
                                        5,
                                        1.2,
                                        0)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = angle * (180/(np.pi / 2))
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Dense oprtical Flow', final)
    if cv2.waitKey(1) == 13:
        break

    frame_gray_init = frame_grey

cap.release()
cv2.destroyAllWindows()
