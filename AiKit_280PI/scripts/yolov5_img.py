import traceback
from multiprocessing import Process, Pipe
import cv2
import numpy as np
import time
import datetime
import threading
import os, sys
import matplotlib.pyplot as plt

from pymycobot.mycobot280 import MyCobot280

IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"  # Seeed adaptatif


class Object_detect():
    global move_finsh

    def __init__(self, camera_x=150, camera_y=10):
        # hériter de la classe parente
        super(Object_detect, self).__init__()

        # déclarer mycobot 280pi
        self.mc = None
        # Angle de mouvement
        self.move_angles = [
            [0.61, 45.87, -92.37, -41.3, 2.02, 9.58],  # initialiser le point
            [18.8, -7.91, -54.49, -23.02, -0.79, -14.76],  # pointer pour saisir
        ]

        # Coordonnées de déplacement
        self.move_coords = [
            [132.2, -136.9, 200.8, -178.24, -3.72, -107.17],  # Zone de tri D
            [238.8, -124.1, 204.3, -169.69, -5.52, -96.52],  # Zone de tri C
            [115.8, 177.3, 210.6, 178.06, -0.92, -6.11],  # Zone de tri A
            [-6.9, 173.2, 201.5, 179.93, 0.63, 33.83],  # Zone de tri B
        ]

        # quel robot : USB* est m5 ; ACM* est wio ; AMA* est raspi
        self.robot_m5 = os.popen("ls /dev/ttyUSB*").readline()[:-1]
        self.robot_wio = os.popen("ls /dev/ttyACM*").readline()[:-1]
        self.robot_raspi = os.popen("ls /dev/ttyAMA*").readline()[:-1]
        self.robot_jes = os.popen("ls /dev/ttyTHS1").readline()[:-1]
        self.raspi = False
        if "dev" in self.robot_m5:
            self.Pin = [2, 5]
        elif "dev" in self.robot_wio:
            # self.Pin = [20, 21]
            self.Pin = [2, 5]

            # for i in self.move_coords:
            #     i[2] -= 20
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
            GPIO.output(20, 1)
            GPIO.output(21, 1)
            self.raspi = True
        if self.raspi:
            self.gpio_status(False)

        # choisir l'endroit pour poser le cube
        self.color = 0
        # paramètres pour calculer les paramètres de découpage de la caméra
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        # définir le cache des coordonnées réelles
        self.cache_x = self.cache_y = 0

        # utiliser pour calculer les coordonnées entre le cube et le mycobot
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        # Les coordonnées du point central de préhension par rapport au mycobot
        self.camera_x, self.camera_y = camera_x, camera_y
        # Les coordonnées du cube par rapport au mycobot
        self.c_x, self.c_y = 0, 0
        # Le rapport des pixels aux valeurs réelles
        self.ratio = 0
        # Obtenir le dictionnaire de marqueurs ArUco qui peut être détecté.
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Obtenir les paramètres du marqueur ArUco.
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # chemin du fichier du modèle yolov5
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modelWeights = self.path + "/scripts/yolov5s.onnx"
        if IS_CV_4:
            self.net = cv2.dnn.readNet(self.modelWeights)
        else:
            print('Le chargement du modèle yolov5 nécessite la version 4 de opencv.')
            exit(0)

        # Constantes.
        self.INPUT_WIDTH = 640  # 640
        self.INPUT_HEIGHT = 640  # 640
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45
        self.CONFIDENCE_THRESHOLD = 0.45

        # Paramètres de texte.
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1

        # Couleurs.
        self.BLACK = (0, 0, 0)
        self.BLUE = (255, 178, 50)
        self.YELLOW = (0, 255, 255)

        '''Charger les noms de classe'''
        classesFile = self.path + "/scripts/coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.change_flag = False

        # contrôle de la pompe pi

    def gpio_status(self, flag):
        if flag:
            self.GPIO.output(20, 0)
            self.GPIO.output(21, 0)
        else:
            self.GPIO.output(20, 1)
            self.GPIO.output(21, 1)

    # démarrer la pompe m5
    def pump_on(self):
        # faire fonctionner la broche 2
        self.mc.set_basic_output(2, 0)
        # faire fonctionner la broche 5
        self.mc.set_basic_output(5, 0)

    # arrêter la pompe m5
    def pump_off(self):
        # arrêter de faire fonctionner la broche 2
        self.mc.set_basic_output(2, 1)
        # arrêter de faire fonctionner la broche 5
        self.mc.set_basic_output(5, 1)

    def check_position(self, data, ids):
        """
        Vérifier de manière répétée si une position est atteinte
        :param data: angle ou coordonnées
        :param ids: angle-0, coordonnées-1
        :return:
        """
        try:
            while True:
                res = self.mc.is_in_position(data, ids)
                # print('res', res)
                if res == 1:
                    time.sleep(0.1)
                    break
                time.sleep(0.1)
        except Exception as e:
            e = traceback.format_exc()
            print(e)

    # Mouvement de préhension
    def move(self, x, y, color):
        print(color)
        # envoyer l'angle pour déplacer mypal260
        self.mc.send_angles(self.move_angles[1], 25)
        self.check_position(self.move_angles[1], 0)

        # envoyer les coordonnées pour déplacer le mycobot
        self.mc.send_coords([x, y, 170.6, 179.87, -3.78, -62.75], 40, 1)  # usb :rx,ry,rz -173.3, -5.48, -57.9
        # self.mc.send_coords([x, y, 150, 179.87, -3.78, -62.75], 25, 0)
        # time.sleep(3)

        self.mc.send_coords([x, y, 65, 179.87, -3.78, -62.75], 40, 1)
        data = [x, y, 65, 179.87, -3.78, -62.75]
        self.check_position(data, 1)

        # ouvrir la pompe
        if "dev" in self.robot_m5 or "dev" in self.robot_wio:
            self.pump_on()
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            self.gpio_status(True)
        time.sleep(1.5)

        tmp = []
        while True:
            if not tmp:
                tmp = self.mc.get_angles()
            else:
                break
        time.sleep(0.5)

        # print(tmp)
        self.mc.send_angles([tmp[0], -0.71, -54.49, -23.02, -0.79, tmp[5]],
                            25)  # [18.8, -7.91, -54.49, -23.02, -0.79, -14.76]
        self.check_position([tmp[0], -0.71, -54.49, -23.02, -0.79, tmp[5]], 0)

        self.mc.send_coords(self.move_coords[color], 40, 1)
        self.check_position(self.move_coords[color], 1)

        # fermer la pompe
        if "dev" in self.robot_m5 or "dev" in self.robot_wio:
            self.pump_off()
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            self.gpio_status(False)
        time.sleep(0.5)

        self.mc.send_angles(self.move_angles[0], 25)
        self.check_position(self.move_angles[0], 0)
        print('Veuillez appuyer sur la barre d\'espace pour ouvrir la caméra pour la prochaine mémorisation et reconnaissance d\'image')
        # print('Please press the space bar to open the camera for the next image storage and recognition')

    # décider de saisir le cube ou non
    def decide_move(self, x, y, color):
        # print(x, y, self.cache_x, self.cache_y)
        # détecter l'état du cube en mouvement ou en cours d'exécution
        # if (abs(x - self.cache_x) + abs(y - self.cache_y)) / 2 > 5:  # mm
        # self.cache_x, self.cache_y = x, y
        # return
        # else:
        self.cache_x = self.cache_y = 0
        # Ajuster la position d'aspiration de la pompe, augmenter y pour se déplacer vers la gauche ; diminuer y pour se déplacer vers la droite ; augmenter x pour avancer ; diminuer x pour reculer

        self.move(x, y, color)

    # initialiser le mycobot
    def run(self):

        if "dev" in self.robot_wio:
            self.mc = MyCobot280(self.robot_wio, 115200)
        elif "dev" in self.robot_m5:
            self.mc = MyCobot280(self.robot_m5, 115200)
        elif "dev" in self.robot_raspi:
            self.mc = MyCobot280(self.robot_raspi, 1000000)
        self.gpio_status(False)
        self.mc.send_angles([0.61, 45.87, -92.37, -41.3, 2.02, 9.58], 20)
        self.check_position([0.61, 45.87, -92.37, -41.3, 2.02, 9.58], 0)

    # dessiner le marqueur aruco
    def draw_marker(self, img, x, y):
        # dessiner un rectangle sur l'image
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # ajouter du texte sur le rectangle
        cv2.putText(
            img,
            "({},{})".format(x, y),
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (243, 0, 0),
            2,
        )

    # obtenir les points de deux aruco
    def get_calculate_params(self, img):
        # Convertir l'image en une image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Détecter le marqueur ArUco.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        """
        Deux Arucos doivent être présents sur l'image et dans le même ordre.
        Il y a deux Arucos dans les Coins, et chaque aruco contient les pixels de ses quatre coins.
        Déterminer le centre de l'aruco par les quatre coins de l'aruco.
        """
        if len(corners) > 0:
            if ids is not None:
                if len(corners) <= 1 or ids[0] == 1:
                    return None
                x1 = x2 = y1 = y2 = 0
                point_11, point_21, point_31, point_41 = corners[0][0]
                x1, y1 = int(
                    (point_11[0] + point_21[0] + point_31[0] + point_41[0]) /
                    4.0), int(
                    (point_11[1] + point_21[1] + point_31[1] + point_41[1])
                    / 4.0)
                point_1, point_2, point_3, point_4 = corners[1][0]
                x2, y2 = int(
                    (point_1[0] + point_2[0] + point_3[0] + point_4[0]) /
                    4.0), int(
                    (point_1[1] + point_2[1] + point_3[1] + point_4[1]) /
                    4.0)
                # print(x1,x2,y1,y2)
                return x1, x2, y1, y2
        return None

    # définir les paramètres de découpage de la caméra
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        print(self.x1, self.y1, self.x2, self.y2)

    # définir les paramètres pour calculer les coordonnées entre le cube et le mycobot
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0 / ratio

    # calculer les coordonnées entre le cube et le mycobot
    def get_position(self, x, y):
        return ((y - self.c_y) * self.ratio +
                self.camera_x), ((x - self.c_x) * self.ratio + self.camera_y)

    """
    Calibrer la caméra en fonction des paramètres de calibration.
    Agrandir le pixel de la vidéo de 1,5 fois, ce qui signifie agrandir la taille de la vidéo de 1,5 fois.
    Si deux valeurs ARuco ont été calculées, couper la vidéo.
    """

    def transform_frame(self, frame):
        # agrandir l'image de 1,5 fois
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0),
                           fx=fx,
                           fy=fy,
                           interpolation=cv2.INTER_CUBIC)
        if self.x1 != self.x2:
            # le rapport de coupe ici est ajusté en fonction de la situation réelle
            frame = frame[int(self.y2 * 0.2):int(self.y1 * 1.15),
                    int(self.x1 * 0.4):int(self.x2 * 1.15)]
        return frame

        '''Dessiner la classe'''

    def draw_label(self, img, label, x, y):
        text_size = cv2.getTextSize(label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(img, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, label, (x, y + dim[1]), self.FONT_FACE, self.FONT_SCALE, self.YELLOW, self.THICKNESS)

    '''
    Pré-traitement
    Prend l'image et le réseau comme paramètres.
    - Premièrement, l'image est convertie en blob. Ensuite, elle est définie comme entrée pour le réseau.
    - La fonction getUnconnectedOutLayersNames() fournit les noms des couches de sortie.
    - Il a les caractéristiques de toutes les couches, l'image se propage à travers ces couches pour obtenir la détection. Après traitement, le résultat de la détection est renvoyé.
    '''

    def pre_process(self, input_image, net):
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.INPUT_HEIGHT, self.INPUT_WIDTH), [0, 0, 0], 1,
                                     crop=False)
        # Définit l'entrée du réseau.
        net.setInput(blob)
        # Exécute la passe avant pour obtenir la sortie des couches de sortie.
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs

    '''Post-traitement
    Filtre les bonnes détections données par le modèle YOLOv5
    Étapes
    - Boucle de détection.
    - Filtre les bonnes détections.
    - Obtient l'index du meilleur score de classe.
    - Rejette les détections dont le score de classe est inférieur au seuil.
    '''

    # détecter l'objet
    def post_process(self, input_image):
        class_ids = []
        confidences = []
        boxes = []
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.INPUT_HEIGHT, self.INPUT_WIDTH), [0, 0, 0], 1,
                                     crop=False)
        # Définit l'entrée du réseau.
        self.net.setInput(blob)
        # Exécute la passe avant pour obtenir la sortie des couches de sortie.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]

        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT
        # point central du pixel
        cx = 0
        cy = 0
        # boucle de détection
        try:
            for r in range(rows):
                row = outputs[0][0][r]
                confidence = row[4]
                if confidence > self.CONFIDENCE_THRESHOLD:
                    classes_scores = row[5:]
                    class_id = np.argmax(classes_scores)
                    if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w / 2) * x_factor)
                        top = int((cy - h / 2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

                        '''Suppression non maximale pour obtenir une seule boîte de délimitation'''
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

                        for i in indices:
                            box = boxes[i]
                            left = box[0]
                            top = box[1]
                            width = box[2]
                            height = box[3]

                            # Dessiner la boîte de délimitation
                            cv2.rectangle(input_image, (left, top), (left + width, top + height), self.BLUE,
                                          3 * self.THICKNESS)

                            # point central du pixel
                            cx = left + (width) // 2
                            cy = top + (height) // 2

                            cv2.circle(input_image, (cx, cy), 5, self.BLUE, 10)

                            # Catégorie détectée                      
                            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])
                            # dessiner l'étiquette de la classe

                            self.draw_label(input_image, label, left, top)

                # cv2.imshow("nput_frame",input_image)
        # return input_image
        except Exception as e:
            print(e)
            exit(0)

        if cx + cy > 0:
            return cx, cy, input_image
        else:
            return None


status = True


def camera_status():
    global status
    status = True
    cap_num = 0
    cap = cv2.VideoCapture(cap_num)


def runs():
    global status

    detect = Object_detect()

    # initialiser le mycobot
    detect.run()

    _init_ = 20  # 
    init_num = 0
    nparams = 0
    num = 0
    real_sx = real_sy = 0

    # chemin de l'image yolov5
    path_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_img = path_dir + '/res/yolov5_detect.png'
    # ouvrir la caméra
    cap_num = 0
    cap = cv2.VideoCapture(cap_num)
    print("*  Raccourcis clavier (à utiliser dans la fenêtre de la caméra) : *")
    print("*  z : prendre une photo (take picture)                    *")
    print("*  q : quitter (quit)                                *")

    while cv2.waitKey(1) < 0:
        if not status:
            cap = cv2.VideoCapture(cap_num)
            status = True
            print("Veuillez placer un objet identifiable dans la fenêtre de la caméra pour la prise de vue")
            print("*  Raccourcis clavier (à utiliser dans la fenêtre de la caméra) : *")
            print("*  z : prendre une photo (take picture)                    *")
            print("*  q : quitter (quit)                                *")
        # lire chaque image
        ret, frame = cap.read()

        cv2.imshow("capture", frame)

        # sauvegarder
        input = cv2.waitKey(1) & 0xFF
        if input == ord('q'):
            print('quitter')
            if cap is not None:
                cap.release()
            break
        if input == ord('z'):
            # Enregistrer l'image
            if frame is not None:
                cv2.imwrite(path_img, frame)
                print("Enregistrer l'image avec succès !")
            else:
                print("La lecture de l'image a échoué, veuillez vérifier si la caméra est connectée")
                continue
            # fermer la caméra
            if cap is not None:
                cap.release()
            status = False

            frame = cv2.imread(path_img)
            # frame = detect.transform_frame(frame)
            # Obtenir les coordonnées des deux ArUcos
            if nparams < 10:
                # Obtenir les coordonnées des deux arucos et les attribuer aux paramètres de découpe
                params = detect.get_calculate_params(frame)
                if params is not None:
                    detect.set_cut_params(params[0], params[1], params[2],
                                          params[3])
                    nparams += 1
                    continue
                else:
                    print("pas d'aruco")
            # Obtenir les coordonnées des deux ArUcos et calculer le rapport des coordonnées du cube
            elif nparams >= 10 and nparams < 20:
                # Obtenir le centre des deux arucos
                params = detect.get_calculate_params(frame)
                # calculer
                if params is not None:
                    detect.sum_x1 += params[0]
                    detect.sum_x2 += params[1]
                    detect.sum_y1 += params[2]
                    detect.sum_y2 += params[3]
                    nparams += 1
                    if nparams == 20:
                        detect.set_params(
                            (detect.sum_x1 + detect.sum_x2) / 20.0,
                            (detect.sum_y1 + detect.sum_y2) / 20.0,
                            abs(detect.sum_x1 - detect.sum_x2) / 10.0)
                        print("aruco trouvé !")
                    continue
                else:
                    print("pas d'aruco")

            else:
                # Détecter le centre du cube
                rect = detect.post_process(frame)
                if rect is not None:
                    # Obtenir les coordonnées réelles du cube par rapport au mycobot
                    real_sx, real_sy, img = rect[0], rect[1], rect[2]
                    detect.draw_marker(img, real_sx, real_sy)

                    # if num > 5:
                    #
                    #     continue

                    # Déterminer la couleur
                    if detect.change_flag:
                        detect.color = detect.color + 1
                        if detect.color > 3:
                            detect.color = 0

                    # 开启灯光
                    if "dev" in detect.robot_raspi or "dev" in detect.robot_jes:
                        if detect.color == 0:  # R
                            detect.mc.set_color(255, 0, 0)
                        elif detect.color == 1:  # G
                            detect.mc.set_color(0, 255, 0)
                        elif detect.color == 2:  # B
                            detect.mc.set_color(0, 0, 255)
                        elif detect.color == 3:
                            detect.mc.set_color(255, 255, 0) # y
                    # 打印颜色
                    color_dict = {
                        0: "rouge",
                        1: "vert",
                        2: "bleu",
                        3: "jaune"
                    }
                    print("La couleur de l'objet identifié est : {}".format(color_dict[detect.color]))
                    print("====================================")
                    # Obtenir les coordonnées réelles
                    real_x, real_y = detect.get_position(
                        real_sx, real_sy)
                    detect.decide_move(real_x, real_y, detect.color)
                    num += 1
                    detect.change_flag = True

                else:
                    print("pas de rectangle")
                print('Reconnaissance terminée, veuillez appuyer sur "q" pour quitter')

    if cap is not None:
        cap.release()
        print('la caméra est fermée')


if __name__ == '__main__':
    runs()
    # try:
    #     runs()
    # except Exception as e:
    #     e = traceback.format_exc()
    #     print(e)
    #     # if cap is not None:
    #     #     cap.release()
    #     #     print('cap is close')
    #     #     cv2.destroyAllWindows()
