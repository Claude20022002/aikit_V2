# AIKit pour ultraArm P340

Ce projet utilise le bras robotique ultraArm P340 avec une caméra pour la reconnaissance et la manipulation d'objets.

## Installation

### Dépendances

Assurez-vous d'avoir Python 3 installé. Vous pouvez ensuite installer les packages nécessaires avec pip :

```bash
pip install opencv-python
pip install pymycobot
pip install numpy
pip install pyserial
```

### Configuration pour Windows

Le code utilise `serial.tools.list_ports.comports()` pour trouver le port série auquel le robot est connecté. Normalement, aucune modification n'est nécessaire car cette fonction est multiplateforme.

Cependant, le script sélectionne souvent le premier port de la liste (`plist[0]`). Si vous avez plusieurs périphériques série connectés, vous devrez peut-être modifier le code pour sélectionner le bon port COM pour l'ultraArm.

Par exemple, dans les fichiers comme `yolov5_img.py`, vous trouverez une ligne comme :
`self.ua = ultraArmP340(self.plist[0], 115200)`
Vous pouvez la remplacer par le port COM correct, par exemple :
`self.ua = ultraArmP340('COM3', 115200)`

## Description des scripts

Voici une description de chaque script Python présent dans le dossier `/scripts` :

-   `yolov5_img.py`: Script principal pour la détection d'objets en utilisant le modèle YOLOv5. Il initialise le bras robotique, calibre la caméra à l'aide de marqueurs ArUco, détecte des objets dans le flux vidéo, et utilise le bras pour les saisir et les déplacer vers des zones de tri prédéfinies.
-   `yolov5s.onnx`: Le modèle de réseau de neurones YOLOv5 pré-entraîné, utilisé pour la détection d'objets.
-   `coco.names`: Fichier contenant les noms des classes d'objets que le modèle YOLOv5 peut détecter.
-   `aikit_shape.py`: Script pour la détection de formes (triangles, carrés, rectangles, cercles).
-   `aikit_img.py`: Similaire à `yolov5_img.py`, utilise la reconnaissance d'image pour déplacer des objets.
-   `aikit_color.py`: Script pour la détection d'objets basée sur leur couleur.
-   `aikit_encode.py`: Script pour la reconnaissance de codes-barres ou QR codes.
-   `add_img.py`: Un script utilitaire pour ajouter de nouvelles images pour l'entraînement ou la reconnaissance.
-   `ultra_execute.py`: Un script qui semble être conçu pour exécuter une série de mouvements ou de tâches avec le bras robotique.
-   `test.py`: Un script de test pour vérifier rapidement la connexion et les mouvements de base du bras robotique.
-   `megaAiKit.py`: Semble être une version du script pour un autre type de carte (Mega) ou un script plus complet.
-   `conveyor_belt_color.py`: Script adapté pour un scénario avec un tapis roulant, triant les objets par couleur.
-   `color_test.py`: Un script de test simple pour la fonctionnalité de reconnaissance de couleur.
-   `OpenVideo.py`: Un script simple pour ouvrir le flux vidéo de la caméra.
-   `__init__.py`: Fichier vide qui indique que le dossier `scripts` est un package Python.
