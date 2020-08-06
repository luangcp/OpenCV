""" O RASTREAMENTO COM CSRT É O MELHOR POREM É MAIS LENTO"""
import cv2  # importando o opencv
import sys  # recursos do sistema operacional
from random import randint  # Grando numeros aleatorios

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # Versões do OpenCV
print(major_ver, major_ver, subminor_ver)

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']   # Rastreadores. ALGORITMOS DISPONIVEIS
tracker_type = tracker_types[6]  # SELECIONANDO O ALGORITIMO NA LISTA COMEÇA EM 0
# print(tracker_types) MOSTRANDO QUAL FOI O ALGORITMO SELECIONADO

# TRABALHANDO COM AS VERSÕES:
if int(minor_ver) < 3:
    tracker = tracker_type
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMEDIANFLOW_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()

# print(tracker)

# Fazer o carregamento do video

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

    cv2.putText(frame, tracker_type + 'Tracker', (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27:  # significa que o ESC INTERROMPE
        break




