import traceback

import cv2
import numpy as np
import time
import os, sys
import math

from pymycobot.mycobot280 import MyCobot280

IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"


# Seeed adaptatif


class Object_detect():

    def __init__(self, camera_x=162, camera_y=15):
        # hériter de la classe parente
        super(Object_detect, self).__init__()
        # déclarer mycobot280
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

        # Initialiser le soustracteur de fond
        self.mog = cv2.bgsegm.createBackgroundSubtractorMOG()

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
        # envoyer l'angle pour déplacer mycobot280
        print(color)
        self.mc.send_angles(self.move_angles[1], 25)
        self.check_position(self.move_angles[1], 0)

        # envoyer les coordonnées pour déplacer le mycobot
        self.mc.send_coords([x, y, 170.6, 179.87, -3.78, -62.75], 40, 1)  # usb :rx,ry,rz -173.3, -5.48, -57.9

        # self.mc.send_coords([x, y, 150, 179.87, -3.78, -62.75], 25, 0)
        # time.sleep(3)

        self.mc.send_coords([x, y, 65.5, 179.87, -3.78, -62.75], 40, 1)
        data = [x, y, 65.5, 179.87, -3.78, -62.75]
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
        time.sleep(5)

        self.mc.send_angles(self.move_angles[0], 25)
        self.check_position(self.move_angles[0], 0)

    # décider de saisir le cube ou non
    def decide_move(self, x, y, color):
        print(x, y, self.cache_x, self.cache_y)
        # détecter l'état du cube en mouvement ou en cours d'exécution
        if (abs(x - self.cache_x) + abs(y - self.cache_y)) / 2 > 5:  # mm
            self.cache_x, self.cache_y = x, y
            return
        else:
            self.cache_x = self.cache_y = 0
            # Ajuster la position d'aspiration de la pompe, augmenter y pour se déplacer vers la gauche ; diminuer y pour se déplacer vers la droite ; augmenter x pour avancer ; diminuer x pour reculer
            self.move(x, y, color)

    # initialiser mycobot280
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
        # dessiner un rectangle sur l'img
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # ajouter du texte sur le rectangle
        cv2.putText(img, "({},{})".format(x, y), (x, y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (243, 0, 0), 2, )

    # obtenir les points de deux aruco
    def get_calculate_params(self, img):
        # Convertir l'image en une image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Détecter le marqueur ArUco.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

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
                x1, y1 = int((point_11[0] + point_21[0] + point_31[0] + point_41[0]) / 4.0), int(
                    (point_11[1] + point_21[1] + point_31[1] + point_41[1]) / 4.0)
                point_1, point_2, point_3, point_4 = corners[1][0]
                x2, y2 = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4.0), int(
                    (point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4.0)
                return x1, x2, y1, y2
        return None

    # définir les paramètres de découpage de la caméra
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        print(self.x1, self.y1, self.x2, self.y2)

    # définir les paramètres pour calculer les coordonnées du cube
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0 / ratio

    # calculer les coordonnées du cube
    def get_position(self, x, y):
        return ((y - self.c_y) * self.ratio + self.camera_x), ((x - self.c_x) * self.ratio + self.camera_y)

    """
    Calibrer la caméra en fonction des paramètres de calibration.
    Agrandir le pixel de la vidéo de 1,5 fois, ce qui signifie agrandir la taille de la vidéo de 1,5 fois.
    Si deux valeurs ARuco ont été calculées, couper la vidéo.
    """

    def transform_frame(self, frame):
        # agrandir l'image de 1,5 fois
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        if self.x1 != self.x2:
            # le rapport de coupe ici est ajusté en fonction de la situation réelle
            frame = frame[int(self.y2 * 0.78):int(self.y1 * 1.1),
                    int(self.x1 * 0.88):int(self.x2 * 1.06)]
        return frame

    # fonction de détection par vision par ordinateur
    # d'abord, supprimer l'arrière-plan
    def shape_detect(self, img):
        fgmask = self.mog.apply(img)
        # Niveaux de gris, binaire
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Localiser le contour de l'objet
        contours, hierarchy = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # Calculer l'aire du contour et estimer la forme
            # Filtrer les petits contours
            if cv2.contourArea(c) < 2000:
                continue
            # Trouver le plus petit rectangle circonscrit
            rect = cv2.minAreaRect(c)
            # Calculer les coordonnées du centre du rectangle
            cx, cy = rect[0]
            # dessiner le rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            return cx, cy

        return None, None


def shape_single():
    # ouvrir la caméra
    cap_num = 0
    cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.open()
    # init mycobot280
    detect = Object_detect()
    detect.run()

    _init_ = 20
    init_num = 0
    nparams = 0
    num = 0
    real_sx = real_sy = 0
    print('Début')
    while cv2.waitKey(1) < 0:
        # lire chaque image
        _, frame = cap.read()
        # retourner l'image
        # frame = cv2.flip(frame, 1)
        # traitement de l'image
        frame = detect.transform_frame(frame)
        # Obtenir les coordonnées des deux ArUcos
        if nparams < 10:
            # Obtenir les coordonnées des deux arucos et les attribuer aux paramètres de découpe
            params = detect.get_calculate_params(frame)
            if params is not None:
                detect.set_cut_params(params[0], params[1], params[2], params[3])
                nparams += 1
                continue
            #
            cv2.imshow("figure", frame)
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
                        abs(detect.sum_x1 - detect.sum_x2) / 10.0,
                    )
                    print("Paramètres de l'appareil photo OK")
                continue
            #
            cv2.imshow("figure", frame)
        else:
            #
            # Détecter le centre du cube
            x, y = detect.shape_detect(frame)
            if x is not None:
                # Obtenir les coordonnées réelles du cube par rapport au mycobot
                real_x, real_y = detect.get_position(x, y)

                #
                color = 0

                #
                if real_y > 30:
                    #
                    color = 2
                elif real_y < -30:
                    #
                    color = 3
                else:
                    color = 1
                detect.move(real_x, real_y, color)
                #
                print('OK')
                #
                break
            #
            cv2.imshow("figure", frame)
    #
    cap.release()
    cv2.destroyAllWindows()


def shape_loop():
    # ouvrir la caméra
    cap_num = 0
    cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.open()
    # init mycobot280
    detect = Object_detect()
    detect.run()

    _init_ = 20
    init_num = 0
    nparams = 0
    num = 0
    real_sx = real_sy = 0
    print('Début')
    # Boucle pour identifier et saisir
    while cv2.waitKey(1) < 0:
        # lire chaque image
        _, frame = cap.read()
        # retourner l'image
        # frame = cv2.flip(frame, 1)
        # traitement de l'image
        frame = detect.transform_frame(frame)
        # Obtenir les coordonnées des deux ArUcos
        if nparams < 10:
            # Obtenir les coordonnées des deux arucos et les attribuer aux paramètres de découpe
            params = detect.get_calculate_params(frame)
            if params is not None:
                detect.set_cut_params(params[0], params[1], params[2], params[3])
                nparams += 1
                continue
            #
            cv2.imshow("figure", frame)
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
                        abs(detect.sum_x1 - detect.sum_x2) / 10.0,
                    )
                    print("Paramètres de l'appareil photo OK")
                continue
            #
            cv2.imshow("figure", frame)
        else:
            #
            x, y = detect.shape_detect(frame)
            if x is not None:
                # Obtenir les coordonnées réelles du cube par rapport au mycobot
                real_x, real_y = detect.get_position(x, y)

                #
                color = 0

                #
                if real_y > 30:
                    #
                    color = 2
                elif real_y < -30:
                    #
                    color = 3
                else:
                    color = 1
                detect.move(real_x, real_y, color)
                #
                print('OK')

            #
            cv2.imshow("figure", frame)
    #
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #
    # shape_single()
    shape_loop()
    pass