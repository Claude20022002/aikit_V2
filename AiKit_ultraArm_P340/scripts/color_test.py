import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# Définition des plages de couleurs HSV
# Bleu
lower_blue = np.array([78, 43, 46])
upper_blue = np.array([110, 255, 255])

# Rouge (en HSV, le rouge est à la limite du spectre)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Vert
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

for i in range(0, 19):
    print(cap.get(i))

while(1):
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masques pour chaque couleur
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Masque rouge (combinaison des deux plages)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Application des masques
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)

    # Affichage des résultats
    cv2.imshow('Original', frame)
    cv2.imshow('Bleu', res_blue)
    cv2.imshow('Rouge', res_red)
    cv2.imshow('Vert', res_green)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q') or key == 27:
        print(frame.shape, ret)
        break

cap.release()
cv2.destroyAllWindows()