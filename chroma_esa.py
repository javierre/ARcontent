import numpy as np
import cv2
import sys
import select
import tty
import termios

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)

rows = 1280
cols = 960
cap = cv2.VideoCapture(0) #use external cam
#cap = cv2.VideoCapture('video.mp4')

ret, frame = cap.read()
print(frame.shape)


#img_back = cv2.imread('fondo3.png')
img_back = cv2.imread('titanosaurus3.png')
background = img_back[0:cols, 0:rows ]

thresh = 230
blur = 3
final = ""
display=1
kernel = np.ones((5,5),np.uint8)

# create window
cv2.namedWindow("chroma", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("chroma", cv2.WND_PROP_FULLSCREEN, cv2.CV_WINDOW_FULLSCREEN)

while(True):



    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame,1,frame)

    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel,chan_a,chan_b = cv2.split(lab_image)

    b_channel = cv2.bitwise_not(chan_b)
    a_channel = cv2.add(chan_a,b_channel)
    a_channel = cv2.bitwise_not(a_channel)
    a_channel_inv = cv2.bitwise_not(a_channel)


    # create mask

    im_bw = cv2.threshold(a_channel_inv, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw = cv2.GaussianBlur(im_bw,(blur,blur),0)


    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
    ret ,im_bw = cv2.threshold(im_bw,thresh,255,cv2.THRESH_BINARY)

    im_bw_inv = cv2.bitwise_not(im_bw)
    im_bw_inv = cv2.GaussianBlur(im_bw_inv,(1,1),0)
    im_bw_inv = cv2.morphologyEx(im_bw_inv, cv2.MORPH_CLOSE, kernel)

    roi = cv2.bitwise_and(background,background,mask = im_bw_inv)
    im = cv2.bitwise_and(frame,frame,mask = im_bw)

    # add mask
    dst = cv2.add(roi,im)

    # Display the resulting frame

    #PIA22107-1024x576.jpg
    #curiosity.jpg
    #mars.jpg

    k = cv2.waitKey(33)
    if k == ord('1'):#1048624: #ord('a'): #0
       img_back = cv2.imread('fondo3.png')
       background = img_back[0:cols, 0:rows ]
    elif k == ord('2'):#11048625: #ord('a'): #1
       img_back = cv2.imread('insectos3.png')
       background = img_back[0:cols, 0:rows ]
    elif k == ord('3'):#11048626:#ord('b'): #2
       img_back = cv2.imread('romeral3.png')
       background = img_back[0:cols, 0:rows ]
    elif k == ord('4'):#11048626:#ord('b'): #2
       img_back = cv2.imread('romeral3.png')
       background = img_back[0:cols, 0:rows ]
    elif k == ord('5'):#11048627:#ord('c'): #3
        img_back = cv2.imread('concavenator3.png')
        background = img_back[0:cols, 0:rows ]
    elif k == ord('6'):#11048628:#ord('d'): #4
        img_back = cv2.imread('Aragosaurus3.png')
        background = img_back[0:cols, 0:rows ]
    elif k == ord('7'):#11048629:#ord('e'): #5
        img_back = cv2.imread('titanosaurus3.png')
        background = img_back[0:cols, 0:rows ]
    elif k == ord('8'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/PIA22107-1024x576.jpg')
        background = img_back[0:cols, 0:rows ]
    elif k == ord('9'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/curiosity.jpg')
        background = img_back[0:cols, 0:rows ]
    elif k == ord('0'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/mars.jpg')
        background = img_back[0:cols, 0:rows ]

    if k == 27: #ESC
        break
    if k == ord("+"):
        thresh+=1
    if k == ord("-"):
        thresh-=1

    if k == ord("."):
        blur*=3
    if k == ord(","):
        if blur > 1:
            blur/=3



    if (display == 1):
        final = dst
    if (display == 2):
        final = im
    if (display == 3):
        final = roi
    if (display == 4):
        final = im_bw
    if (display == 5):
        final = im_bw_inv
    if (display == 6):
        final = chan_a
    if (display == 7):
        final = chan_b
    if (display == 8):
        final = l_channel

    #display image
    cv2.putText(dst,"Thresh (+/-): " + str(thresh), (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.5, 0)
    cv2.putText(dst,"Blur (./,): " + str(blur), (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.5, 0)
    cv2.imshow('chroma', final)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
