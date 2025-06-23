import traceback
from multiprocessing import Process, Pipe
import cv2
import numpy as np
import time
import datetime
import threading
import os,sys
import matplotlib.pyplot as plt 
import platform

from pymycobot.mecharm270 import MechArm270

IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"  # Adaptive seeed


class Object_detect():

    def __init__(self, camera_x = 150, camera_y = 7):
        # hérite de la classe parente
        super(Object_detect, self).__init__()

        # déclaration du bras MechArm270 Pi
        self.mc = None
        # Angles de déplacement
        self.move_angles = [
            [0, 0, 0, 0, 90, 0],  # position initiale
            [-33.31, 2.02, -10.72, -0.08, 95, -54.84],  # position pour saisir
        ]

        # Coordonnées de déplacement
        self.move_coords = [
            [96.5, -101.9, 185.6, 155.25, 19.14, 75.88],  # Zone de tri D
            [180.9, -99.3, 184.6, 124.4, 30.9, 80.58],    # Zone de tri C
            [77.4, 122.1, 179.2, 151.66, 17.94, 178.24],  # Zone de tri A
            [2.2, 128.5, 171.6, 163.27, 10.58, -147.25],  # Zone de tri B
        ]
        
        # quel robot : USB* = m5 ; ACM* = wio ; AMA* = raspi
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

            # pour i dans self.move_coords :
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
   
        # choix de la zone de dépôt du cube
        self.color = 0
        # paramètres pour calculer les paramètres de découpe de la caméra
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        # cache des coordonnées réelles
        self.cache_x = self.cache_y = 0

        # utilisé pour calculer la coordonnée entre le cube et le robot
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        # Coordonnées du centre de saisie par rapport au robot
        self.camera_x, self.camera_y = camera_x, camera_y
        # Coordonnées du cube par rapport au robot
        self.c_x, self.c_y = 0, 0
        # Ratio pixel/valeur réelle
        self.ratio = 0
        # Récupère le dictionnaire des marqueurs ArUco détectables
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Récupère les paramètres des marqueurs ArUco
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # chemin du modèle yolov5
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modelWeights = self.path + "/scripts/yolov5s.onnx"
        if IS_CV_4:
            self.net = cv2.dnn.readNet(self.modelWeights)
        else:
            print('Le chargement du modèle yolov5 nécessite la version 4 d\'opencv.')
            exit(0)
            
        # Constantes
        self.INPUT_WIDTH = 640   # 640
        self.INPUT_HEIGHT = 640  # 640
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45 
        self.CONFIDENCE_THRESHOLD = 0.45
        
        # Paramètres du texte
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1
        
        # Couleurs
        self.BLACK  = (0,0,0)
        self.BLUE   = (255,178,50)
        self.YELLOW = (0,255,255)
        
        '''Chargement des noms de classes'''
        classesFile = self.path + "/scripts/coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
       

    # Contrôle de la pompe pour Raspberry Pi
    def gpio_status(self, flag):
        if flag:
            self.GPIO.output(20, 0)
            self.GPIO.output(21, 0)
        else:
            self.GPIO.output(20, 1)
            self.GPIO.output(21, 1)

    # Allumer la pompe (m5)
    def pump_on(self):
        # Active la sortie 2
        self.mc.set_basic_output(2, 0)
        # Active la sortie 5
        self.mc.set_basic_output(5, 0)

    # Éteindre la pompe (m5)
    def pump_off(self):
        # Désactive la sortie 2
        self.mc.set_basic_output(2, 1)
        # Désactive la sortie 5
        self.mc.set_basic_output(5, 1)

    def check_position(self, data, ids):
        """
        Boucle pour vérifier si le robot est arrivé à une position
        :param data: angle ou coordonnées
        :param ids: 0 pour angle, 1 pour coordonnées
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

    # Mouvement de saisie
    def move(self, x, y, color):
        print(color)
        # Envoie les angles pour déplacer le bras
        self.mc.send_angles(self.move_angles[0], 50)
        self.check_position(self.move_angles[0], 0)

        # Envoie les coordonnées pour déplacer le robot
        self.mc.send_coords([x, y, 150, -176.1, 2.4, -125.1], 40, 1) # usb :rx,ry,rz -173.3, -5.48, -57.9

        # self.mc.send_coords([x, y, 150, 179.87, -3.78, -62.75], 25, 0)
        # time.sleep(3)

        # self.mc.send_coords([x, y, 105, 179.87, -3.78, -62.75], 25, 0)
        self.mc.send_coords([x, y, 70, -176.1, 2.4, -125.1], 40, 1)
        
        self.check_position([x, y, 70, -176.1, 2.4, -125.1], 1)

        # Allume la pompe
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
        self.mc.send_angles([tmp[0], 17.22, -32.51, tmp[3], 97, tmp[5]],30) # [18.8, -7.91, -54.49, -23.02, -0.79, -14.76]
        self.check_position([tmp[0], 17.22, -32.51, tmp[3], 97, tmp[5]], 0)

        self.mc.send_coords(self.move_coords[color], 40, 1)
        self.check_position(self.move_coords[color], 1)

        # Éteint la pompe
        if "dev" in self.robot_m5 or "dev" in self.robot_wio:
            self.pump_off()
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            self.gpio_status(False)
        time.sleep(5)

        self.mc.send_angles(self.move_angles[1], 50)
        self.check_position(self.move_angles[1], 0)

        print('请按空格键打开摄像头进行下一次图像存储和识别')
        print('Veuillez appuyer sur la barre d\'espace pour ouvrir la caméra pour le prochain stockage et reconnaissance d\'image')

    # Décider si on saisit le cube
    def decide_move(self, x, y, color):
        # print(x, y, self.cache_x, self.cache_y)
        # détecte si le cube bouge ou non
        #if (abs(x - self.cache_x) + abs(y - self.cache_y)) / 2 > 5:  # mm
            #self.cache_x, self.cache_y = x, y
            #return
        #else:
        self.cache_x = self.cache_y = 0
        # Ajuste la position d'aspiration : y augmente = déplacement à gauche ; y diminue = déplacement à droite ; x augmente = déplacement vers l'avant ; x diminue = déplacement vers l'arrière
        self.move(x, y, color)
      

    # Initialisation du robot
    def run(self):
    
        if "dev" in self.robot_wio :
            self.mc = MechArm270(self.robot_wio, 115200)
        elif "dev" in self.robot_m5:
            self.mc = MechArm270(self.robot_m5, 115200)
        elif "dev" in self.robot_raspi:
            self.mc = MechArm270(self.robot_raspi, 1000000)
        self.gpio_status(False)
        self.mc.send_angles([-33.31, 2.02, -10.72, -0.08, 95, -54.84], 50)
        time.sleep(3)


    # Dessiner un marqueur aruco
    def draw_marker(self, img, x, y):
        # dessine un rectangle sur l'image
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # ajoute du texte sur le rectangle
        cv2.putText(
            img,
            "({},{})".format(x, y),
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (243, 0, 0),
            2,
        )

    # Récupère les points des deux aruco
    def get_calculate_params(self, img):
        # Convertit l'image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Détecte les marqueurs ArUco
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        """
        Deux Aruco doivent être présents sur l'image et dans le même ordre.
        Il y a deux Aruco dans corners, et chaque aruco contient les pixels de ses quatre coins.
        Détermine le centre de l'aruco à partir de ses quatre coins.
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
                #print(x1,x2,y1,y2)
                return x1, x2, y1, y2
        return None

    # Définit les paramètres de découpe de la caméra
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        print(self.x1, self.y1, self.x2, self.y2)

    # Définit les paramètres pour calculer la coordonnée entre le cube et le robot
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0 / ratio

    # Calcule la coordonnée entre le cube et le robot
    def get_position(self, x, y):
        return ((y - self.c_y) * self.ratio +
                self.camera_x), ((x - self.c_x) * self.ratio + self.camera_y)

    """
    Calibre la caméra selon les paramètres de calibration.
    Agrandit l'image vidéo de 1,5 fois.
    Si deux valeurs ArUco ont été calculées, découpe la vidéo.
    """

    def transform_frame(self, frame):
        # agrandit l'image de 1,5 fois
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0),
                           fx=fx,
                           fy=fy,
                           interpolation=cv2.INTER_CUBIC)
        if self.x1 != self.x2:
            # le ratio de découpe ici est ajusté selon la situation réelle
            frame = frame[int(self.y2 * 0.2):int(self.y1 * 1.15),
                          int(self.x1 * 0.4):int(self.x2 * 1.15)]
        return frame

        '''Dessine la classe détectée'''
    def draw_label(self,img,label,x,y):
        text_size = cv2.getTextSize(label,self.FONT_FACE,self.FONT_SCALE,self.THICKNESS)
        dim,baseline = text_size[0],text_size[1]
        cv2.rectangle(img,(x,y),(x+dim[0],y+dim[1]+baseline),(0,0,0),cv2.FILLED)
        cv2.putText(img,label,(x,y+dim[1]),self.FONT_FACE,self.FONT_SCALE,self.YELLOW,self.THICKNESS)

    '''
    Prétraitement
    Prend l'image et le réseau comme paramètres.
    - L'image est convertie en blob puis définie comme entrée du réseau.
    - La fonction getUnconnectedOutLayerNames() fournit les noms des couches de sortie.
    - L'image passe à travers toutes les couches pour obtenir les détections.
    - Retourne les résultats de détection.
    '''
    def pre_process(self,input_image,net):
        blob = cv2.dnn.blobFromImage(input_image,1/255,(self.INPUT_HEIGHT,self.INPUT_WIDTH),[0,0,0], 1, crop=False)
        # Définit l'entrée du réseau
        net.setInput(blob)
        # Passe avant pour obtenir la sortie des couches de sortie
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs
    '''Post-traitement
    Filtre les bonnes détections du modèle YOLOv5
    Étapes :
    - Boucle sur les détections.
    - Filtre les bonnes détections.
    - Récupère l'indice du score de classe le plus élevé.
    - Ignore les détections dont le score de classe est trop faible.
    '''
    
    # Détection d'objet
    def post_process(self,input_image):
        class_ids = []
        confidences = []
        boxes = []
        blob = cv2.dnn.blobFromImage(input_image,1/255,(self.INPUT_HEIGHT,self.INPUT_WIDTH),[0,0,0], 1, crop=False)
        # Définit l'entrée du réseau
        self.net.setInput(blob)
        # Passe avant pour obtenir la sortie des couches de sortie
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        rows = outputs[0].shape[1]
        image_height ,image_width = input_image.shape[:2]
        
        x_factor = image_width/self.INPUT_WIDTH
        y_factor = image_height/self.INPUT_HEIGHT
        # Centre du pixel
        cx = 0 
        cy = 0 
        # Boucle sur les détections
        try:
            for r in range(rows):
                row = outputs[0][0][r]
                confidence = row[4]
                if confidence>self.CONFIDENCE_THRESHOLD:
                    classes_scores = row[5:]
                    class_id = np.argmax(classes_scores)
                    if (classes_scores[class_id]>self.SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx,cy,w,h = row[0],row[1],row[2],row[3]
                        left = int((cx-w/2)*x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
                        
                        '''Suppression non maximale pour obtenir une boîte standard'''
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
                       
                        for i in indices:
                            box = boxes[i]
                            left = box[0]
                            top = box[1]
                            width = box[2]
                            height = box[3]
                                    
                            # Dessine la boîte standard
                            cv2.rectangle(input_image, (left, top), (left + width, top + height),self.BLUE, 3*self.THICKNESS)
                           
                            # Centre du pixel
                            cx = left+(width)//2 
                            cy = top +(height)//2
                           
                            cv2.circle(input_image, (cx,cy),  5,self.BLUE, 10)
                          
                            # Classe détectée                     
                            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])             
                            # Dessine la classe
                            self.draw_label(input_image, label, left, top)
                            
                #cv2.imshow("nput_frame",input_image)
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

    # init mycobot
    detect.run()

    _init_ = 20  # 
    init_num = 0
    nparams = 0
    # num = 0
    # real_sx = real_sy = 0
    
    # yolov5 img path
    path_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_img = path_dir + '/res/yolov5_detect.png'
    # open the camera
    if platform.system() == "Windows":
        cap_num = 1
    elif platform.system() == "Linux":
        cap_num = 0
    cap = cv2.VideoCapture(cap_num)
    
    print("*  热键(请在摄像头的窗口使用):                   *")
    print("*  hotkey(please use it in the camera window): *")
    print("*  z: 拍摄图片(take picture)                    *")
    print("*  q: 退出(quit)                                *")

    while cv2.waitKey(1)<0:
        if not status:
            cap = cv2.VideoCapture(cap_num)
            status = True
            print("请将可识别物体放置摄像头窗口进行拍摄")
            print("Please place an identifiable object in the camera window for shooting")
            print("*  热键(请在摄像头的窗口使用):                   *")
            print("*  hotkey(please use it in the camera window): *")
            print("*  z: 拍摄图片(take picture)                    *")
            print("*  q: 退出(quit)                                *")
        # 读入每一帧
        ret, frame = cap.read()

        cv2.imshow("capture", frame)
                                      

            # 存储
        input = cv2.waitKey(1) & 0xFF
        if input == ord('q'):
            print('quit')
            break
        elif input == ord('z'):
            print("请截取白色识别区域的部分")
            print("Please capture the part of the white recognition area")
            # 选择ROI
            roi = cv2.selectROI(windowName="capture",
                        img=frame,
                        showCrosshair=False,
                        fromCenter=False)
            x, y, w, h = roi
            print(roi)
            if roi != (0, 0, 0, 0):
                crop = frame[y:y+h, x:x+w]
                cv2.imwrite(path_img, crop)
                cap.release()
                cv2.destroyAllWindows()
                status=False
            
            
            while True:
                frame = cv2.imread(path_img)
                
                #frame = frame[170:700, 230:720]
                frame = detect.transform_frame(frame)
                
                # cv2.imshow('oringal',frame)
                
                if _init_ > 0:
                    _init_-=1
                    continue
                # calculate the parameters of camera clipping
                if init_num < 20:
                    if detect.get_calculate_params(frame) is None:
                        cv2.imshow("figure",frame)
                        continue
                    else:
                        x1,x2,y1,y2 = detect.get_calculate_params(frame)
                        detect.draw_marker(frame,x1,y1)
                        detect.draw_marker(frame,x2,y2)
                        detect.sum_x1+=x1
                        detect.sum_x2+=x2
                        detect.sum_y1+=y1
                        detect.sum_y2+=y2
                        init_num+=1
                        continue
                elif init_num==20:
                    detect.set_cut_params(
                        (detect.sum_x1)/20.0, 
                        (detect.sum_y1)/20.0, 
                        (detect.sum_x2)/20.0, 
                        (detect.sum_y2)/20.0, 
                    )
                    detect.sum_x1 = detect.sum_x2 = detect.sum_y1 = detect.sum_y2 = 0
                    init_num+=1
                    continue

                # calculate params of the coords between cube and mycobot
                if nparams < 10:
                    if detect.get_calculate_params(frame) is None:
                        cv2.imshow("figure",frame)
                        continue
                    else:
                        x1,x2,y1,y2 = detect.get_calculate_params(frame)
                        detect.draw_marker(frame,x1,y1)
                        detect.draw_marker(frame,x2,y2)
                        detect.sum_x1+=x1
                        detect.sum_x2+=x2
                        detect.sum_y1+=y1
                        detect.sum_y2+=y2
                        nparams+=1
                        continue
                elif nparams==10:
                    nparams+=1
                    # calculate and set params of calculating real coord between cube and mycobot
                    detect.set_params(
                        (detect.sum_x1+detect.sum_x2)/20.0, 
                        (detect.sum_y1+detect.sum_y2)/20.0, 
                        abs(detect.sum_x1-detect.sum_x2)/10.0+abs(detect.sum_y1-detect.sum_y2)/10.0
                    )
                    print('start yolov5 recognition.....')
                    print("ok") 
                    continue
                # yolov5 detect result        
                detect_result = detect.post_process(frame)
                print('pick...')
                if detect_result:
                    x, y, input_img = detect_result
                    
                    real_x, real_y = detect.get_position(x, y)
                    print("real_x,real_y:", (round(real_x, 2), round(real_y, 2)))
                  
                    a = threading.Thread(target=lambda:detect.decide_move(real_x, real_y, detect.color))
                    a.start()
                  
                    cv2.imshow("detect_done",input_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                break
                

if __name__ == "__main__":

    runs()


































