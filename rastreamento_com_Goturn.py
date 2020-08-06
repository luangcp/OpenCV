# Não é muito bom
import cv2, sys, os
from random import randint

if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    print('Erro ao carregar os arquivos do Goturn')
    sys.exit()

tracker = cv2.TrackerGOTURN_create()

video = cv2.VideoCapture('videos/race.mp4')
if not video.isOpened():
    print('Não foi possivel carregar o video')
    sys.exit()  # saindo da leitura
# fazendo a leitura do primeiro frame

ok, frame = video.read()
if not ok:
    print('Não foi possivel ler o arquivo de video')
    sys.exit()  # saindo da leitura
# print(ok)

# bordin box
bbox = cv2.selectROI(frame, False)  # FAZENDO A SELEÇÃO DA REGIÃO DE INTERESSE
# print(bbox)  # valores e posições do bordin box
# Inicializar o tracker com o primeiro frame com a região do objeto

ok = tracker.init(frame, bbox)
# print(ok) se retornar TRUE ele conseguiu fazer a inicialização

colors = (randint(0, 255), randint(0, 255), randint(0, 255))
print(colors)

""" 
OBS: Quando trabalhando com openCV, o padrão RGB é representado ao contrario, sendo BGR

"""

# criando um loop infinito pra ele percorrer todos os frames do video

while True:
    ok, frame = video.read()
    if not ok:  # quando terminar o video ele para
        break
    # FPS -> FRAMES POR SEGUNDOS
    # MEDINDO O FPS:
    timer = cv2.getTickCount()  # retorna o numero de ciclos de relogio (ciclos de clock)
    ok, bbox = tracker.update(frame)  # indica em qual posição esta no frame atual, mudança de posição do frame

    # print(ok, bbox) # mostrando a posição

    # calculo do fps
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, 'Falha no rastreamento', (108, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, 'Goturn Tracker', (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27:  # significa que o ESC INTERROMPE
        break

