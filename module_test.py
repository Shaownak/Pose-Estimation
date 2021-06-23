import cv2 as cv
import time
import PoseModule as pm


cap = cv.VideoCapture('Videos/video1.mp4')
pTime = 0
detector = pm.PoseDetector()

while True:
    success, imgo = cap.read()
    img = cv.resize(imgo, (620, 360))
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Image", img)

    cv.waitKey(1)
