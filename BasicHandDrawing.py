import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()

    if not success:
        break

    imgrgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results = hands.process(imgrgb)
    if results.multi_hand_landmarks:
        for handlmk in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlmk,mpHands.HAND_CONNECTIONS)

    cv.imshow("Image",img)
    if cv.waitKey(1)== ord('q'):
        break
cap.release()
cv.destroyAllWindows()