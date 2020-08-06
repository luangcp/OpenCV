"""A ideia é o proprio detector detecta as pessoas faz o bordin box e aciona
o rastreador. Ambos trabalhando juntos"""

import cv2  # Importando o openCV
import sys  # importando complementos
from random import randint  # importando numeros aleatorios

tracker = cv2.TrackerCSRT_create()  # Escolhendo o rastreador

video = cv2.VideoCapture("videos/walking.avi")  # Abrindo o video

if not video.isOpened():  # se o video não abrir
    print("Não foi possivel abrir o video")
    sys.exit()

ok, frame = video.read()  # Buscar o primeiro frame
if not ok:  # Se não achar o primeiro frame
    print('Não é possivel ler o arquivo de vídeo')
    sys.exit()

cascade = cv2.CascadeClassifier('cascade/fullbody.xml')  # Detector


def detectar():  # Função do detector
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # COLOCANDO EM ESCALA DE CINZA
        detection = cascade.detectMultiScale(frame_gray)
        for (x, y, l, a) in detection:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0,0,255), 2)
            cv2.imshow("Detecção", frame)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if x > 0:
                print('Detecção efetuada pelo haarcascade')
                return x, y, l, a


bbox = detectar()
# print(bbox)


ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)  # Atualizando o frame
    if ok:  # se ele conseguiu fazer o rastreamento
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:  # Se não conseguir fazer o rastreamento
        print('Falha no rastreamento. Será executado o detector haarcascade')
        # se tiver dado erro executar o detector novamente
        bbox = detectar()
        tracker = cv2.TrackerMOSSE_create()
        tracker.init(frame, bbox)

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break





