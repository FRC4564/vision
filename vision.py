import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(min_detection_confidence=.7,)

while cv2.waitKey(1) == -1:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB.flags.writeable = False   #to imporve performance
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        # for each hand detected 
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # extract lankmarks of a hand
            # Reference: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
            #for id, lm in enumerate(handLms.landmark):
            #    print(id, lm.x, lm.y, lm.z)
            wrist = handLms.landmark[0]  
            h, w, c = img.shape
            x  = int(wrist.x*w)
            y  = int(wrist.y*h)
            cv2.circle(img,(x,y), 5,(255,0,0),cv2.FILLED)

    cv2.imshow("Image", img)
cap.release()
 