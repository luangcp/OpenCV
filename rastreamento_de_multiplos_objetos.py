"""RASTREAMENTO DE MULTIPLOS OBJETOS"""
import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

# VAMOS FAZER UMA FUNÇÃO


def createTrackerByName(trackerType):
    if trackerType == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == tracker_types[4]:
        tracker = cv2.TrackerMEDIANFLOW_create()
    elif trackerType == tracker_types[5]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == tracker_types[6]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Nome incorreto')
        print('Os rastreadores disponiveis são:')
        for t in tracker_types:
            print(t)

    return tracker


cap = cv2.VideoCapture("videos/race.mp4")

ok, frame = cap.read()  # Fazendo a leitura do primeiro frame do video, captando a imagem
if not ok:
    print('Não é possivel ler o arquivo de video')
    sys.exit(1)


bboxes = []  # Variavel vazia, assim pode-se selecionar varios
colors = []  # também com color


# Criando o loop

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))  # LEMBRAR QUE É BGR E N RGB
    print('Precione Q para sair das caixas de seleção e começar a rastrear')
    print('Pressione qualquer outra tecla pasa selecionar o proximo objeto')
    k = cv2.waitKey(0) & 0XFF  # ta esperando uma tecla
    if k == 113:
        break

print('Caixas delimitadoras selecionadas {}'.format(bboxes))
print('Cores {}'.format(colors))

"--------------------------------------------------------"
trackertype = 'CSRT'  # DEFININDO O ALGORITMO UTILIZADO
"--------------------------------------------------------"

multiTracker = cv2.MultiTracker_create()

for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackertype), frame, bbox)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    ok, boxes = multiTracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2, 1)

    cv2.imshow('MultiTracker', frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break
