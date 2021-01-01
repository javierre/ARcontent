import cv2

cv2.namedWindow("preview")

cap = cv2.VideoCapture(0)
#internal camera
#cap = cv2.VideoCapture(0)
#external
#cap = cv2.VideoCapture(1)

if cap.isOpened(): # try to get the first frame
    ret, frame = cap.read()
else:
    ret = False

cv2.namedWindow("chroma")

while(ret):
    ret, frame = cap.read()
    key = cv2.waitKey(20)
    cv2.imshow('chroma', frame)
    if key == 27: # exit on ESC
        break    
cv2.destroyWindow("preview")
