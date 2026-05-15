import cv2 as cv
import mediapipe as mp
import time 

cap= cv.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

ctime = 0
ptime=0

while True:
    success,img = cap.read()

    if not success:
        break

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlmk in results.multi_hand_landmarks:
            for id,lm in enumerate(handlmk.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,lm)

                if(id==8):
                    cv.circle(img,(cx,cy),15,(0,255,0),cv.FILLED)
            mpDraw.draw_landmarks(img,handlmk,mphands.HAND_CONNECTIONS)

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3.0,(255,0,255),2)

    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()