import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture('Videos/video1.mp4')
pTime = 0


while True:
    success, imgo = cap.read()
    img = cv.resize(imgo, (620,360))

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv.imshow("Image", img)
    
    cv.waitKey(1)
