from multiprocessing import Process, Pipe
import cv2
import numpy as np
import time
import datetime
import threading
import os,sys
import serial
import serial.tools.list_ports
import pymycobot
from packaging import version
# Version minimale requise
MAX_REQUIRE_VERSION = '3.9.1'
current_verison = pymycobot.__version__
print('Version actuelle de la bibliothèque pymycobot : {}'.format(current_verison))
if version.parse(current_verison) > version.parse(MAX_REQUIRE_VERSION):
    from pymycobot.ultraArmP340 import ultraArmP340
    class_name = 'new'
else:
    from pymycobot.ultraArm import ultraArm
    class_name = 'old'
    print("Note : Cette classe n'est plus maintenue depuis la v3.6.0, veuillez vous référer à la documentation du projet : https://github.com/elephantrobotics/pymycobot/blob/main/README.md")


from pymycobot.mycobot import MyCobot

IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"  # Seeed adaptatif


class Object_detect():

    def __init__(self, camera_x = 255, camera_y = -10):
        # hériter de la classe parente
        super(Object_detect, self).__init__()

        # déclarer ultraArm P340
        self.ua = None
        
        # obtenir le port série réel
        self.plist = [
        str(x).split(" - ")[0].strip() for x in serial.tools.list_ports.comports()
    ]
        
        # Angle de mouvement
        self.move_angles = [
            [0.0, 0.0, 0.0],  # initialiser le point
            # [19.48, 0.0, 0.0],  # point pour saisir
            [25.55, 0.0, 15.24],
            [0.0, 14.32, 0.0],  # point pour saisir
        ]

        # Coordonnées de déplacement
        self.move_coords = [
            [141.53, 148.67, 43.73], # Zone de tri D
            [248.52, 152.35, 53.45],    # Zone de tri C
            [269.02, -161.65, 51.42],   # Zone de tri A
            [146.8, -159.53, 50.44],     # Zone de tri B
        ]
   
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
        self.INPUT_WIDTH = 640   # 640
        self.INPUT_HEIGHT = 640  # 640
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45 
        self.CONFIDENCE_THRESHOLD = 0.45
        
        # Paramètres de texte.
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1
        
        # Couleurs.
        self.BLACK  = (0,0,0)
        self.BLUE   = (255,178,50)
        self.YELLOW = (0,255,255)
        
        '''Charger les noms de classe'''
        classesFile = self.path + "/scripts/coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
       

    # démarrer la pompe
    def pump_on(self):
        self.ua.set_gpio_state(0)

    # arrêter la pompe
    def pump_off(self):
        self.ua.set_gpio_state(1)

     # Mouvement de préhension
    def move(self, x, y, color):
        # envoyer l'angle pour déplacer ultraArm P340
        self.ua.set_angles(self.move_angles[2], 50)
        time.sleep(3)
        
        # envoyer les coordonnées pour déplacer ultraArm P340. Ajuster la hauteur de la pompe en fonction des différents bras robotiques de la plaque de base
        self.ua.set_coords([x, -y, 65.51], 50)
        time.sleep(1.5)
        self.ua.set_coords([x, -y, -70], 50)
        time.sleep(2)

        # ouvrir la pompe
        self.pump_on()
        time.sleep(1.5)
        self.ua.set_angles(self.move_angles[0], 50)
        # self.ua.set_angle(2, 0, 50)
        # time.sleep(0.02)
        # self.ua.set_angle(3, 0, 50)
        time.sleep(0.5)

        self.ua.set_coords(self.move_coords[color], 50)
    
        time.sleep(7)

        # fermer la pompe
   
        self.pump_off()
        time.sleep(8)

        self.ua.set_angles(self.move_angles[1], 50)
        time.sleep(1.5)

        print('Veuillez appuyer sur la barre d\'espace pour ouvrir la caméra pour la prochaine mémorisation et reconnaissance d\'image')

    # décider de saisir le cube ou non
    def decide_move(self, x, y, color):
        # print(x, y, self.cache_x, self.cache_y)
        # détecter l'état du cube en mouvement ou en cours d'exécution
        #if (abs(x - self.cache_x) + abs(y - self.cache_y)) / 2 > 5:  # mm
            #self.cache_x, self.cache_y = x, y
            #return
        #else:
        self.cache_x = self.cache_y = 0
        # Ajuster la position d'aspiration de la pompe, augmenter y pour se déplacer vers la gauche ; diminuer y pour se déplacer vers la droite ; augmenter x pour avancer ; diminuer x pour reculer
   
        self.move(x, y, color)
      

    # initialiser ultraArm P340
    def run(self):
        if class_name == 'old':
            self.ua = ultraArm(self.plist[0], 115200)
        else:
            self.ua = ultraArmP340(self.plist[0], 115200)
        self.ua.go_zero()
        self.ua.set_angles([25.55, 0.0, 15.24], 50)
        time.sleep(3)


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
                #print(x1,x2,y1,y2)
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
    def draw_label(self,img,label,x,y):
        text_size = cv2.getTextSize(label,self.FONT_FACE,self.FONT_SCALE,self.THICKNESS)
        dim,baseline = text_size[0],text_size[1]
        cv2.rectangle(img,(x,y),(x+dim[0],y+dim[1]+baseline),(0,0,0),cv2.FILLED)
        cv2.putText(img,label,(x,y+dim[1]),self.FONT_FACE,self.FONT_SCALE,self.YELLOW,self.THICKNESS)

    '''
    Pré-traitement
    Prend l'image et le réseau comme paramètres.
    - Premièrement, l'image est convertie en blob. Ensuite, elle est définie comme entrée pour le réseau.
    - La fonction getUnconnectedOutLayersNames() fournit les noms des couches de sortie.
    - Il a les caractéristiques de toutes les couches, l'image se propage à travers ces couches pour obtenir la détection. Après traitement, le résultat de la détection est renvoyé.
    '''
    def pre_process(self,input_image,net):
        blob = cv2.dnn.blobFromImage(input_image,1/255,(self.INPUT_HEIGHT,self.INPUT_WIDTH),[0,0,0], 1, crop=False)
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
    def post_process(self,input_image):
        class_ids = []
        confidences = []
        boxes = []
        blob = cv2.dnn.blobFromImage(input_image,1/255,(self.INPUT_HEIGHT,self.INPUT_WIDTH),[0,0,0], 1, crop=False)
        # Définit l'entrée du réseau.
        self.net.setInput(blob)
        # Exécute la passe avant pour obtenir la sortie des couches de sortie.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        rows = outputs[0].shape[1]
        image_height ,image_width = input_image.shape[:2]
        
        x_factor = image_width/self.INPUT_WIDTH
        y_factor = image_height/self.INPUT_HEIGHT
        # point central du pixel
        cx = 0 
        cy = 0 
        # boucle de détection
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
                        
                        

                        '''Suppression non maximale pour obtenir une seule boîte de délimitation'''
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
                       
                        for i in indices:
                            box = boxes[i]
                            left = box[0]
                            top = box[1]
                            width = box[2]
                            height = box[3]
                                    
                            # Dessiner la boîte de délimitation
                            cv2.rectangle(input_image, (left, top), (left + width, top + height),self.BLUE, 3*self.THICKNESS)
                           
                            # point central du pixel
                            cx = left+(width)//2 
                            cy = top +(height)//2
                           
                            cv2.circle(input_image, (cx,cy),  5,self.BLUE, 10)
                          
                            

                            # Catégorie détectée                     
                            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])             
                            # dessiner l'étiquette de la classe
                             
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

    # initialiser ultraArm
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

    while cv2.waitKey(1)<0:
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
            break
        elif input == ord('z'):
            print("Veuillez découper la partie de la zone de reconnaissance blanche")
            # Sélectionner ROI
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
                # calculer les paramètres de découpage de la caméra
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

                # calculer les paramètres des coordonnées entre le cube et mycobot
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
                    # calculer et définir les paramètres de calcul des coordonnées réelles entre le cube et mycobot
                    detect.set_params(
                        (detect.sum_x1+detect.sum_x2)/20.0, 
                        (detect.sum_y1+detect.sum_y2)/20.0, 
                        abs(detect.sum_x1-detect.sum_x2)/10.0+abs(detect.sum_y1-detect.sum_y2)/10.0
                    )
                    print('démarrage de la reconnaissance yolov5.....')
                    print("d'accord") 
                    continue
                # résultat de détection yolov5        
                detect_result = detect.post_process(frame)
                print('collecte...')
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
    

































