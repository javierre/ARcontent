import numpy as np
import cv2
import moviepy.editor as mp
import os

#default values
cols = 1280
rows = 960

#select mode: image video or camera
mode="camera"

if mode=="camera":
    cap = cv2.VideoCapture(0) #0 for internal camera, 1 for external
    #cap = cv2.VideoCapture(0) #use internal cam
    if cap.isOpened(): # try to get the first frame
        ret, frame = cap.read()
        cols=frame.shape[1]
        rows=frame.shape[0]
    else:
        ret = False

elif mode=="video":
    myvideo="esa/example.mp4"
    videonuevo="nuevovideo.mp4"
    cap = cv2.VideoCapture(myvideo) #videofile
    video = mp.VideoFileClip(myvideo)
    audio = video.audio
    #player = MediaPlayer(video_path) #audio
    if cap.isOpened(): # try to get the first frame
        ret, frame = cap.read()
        cols=frame.shape[1]
        rows=frame.shape[0]
    else:
        ret = False

elif mode=="image":
    frame = cv2.imread('esa/esashow.jpg')
    cols=frame.shape[1]
    rows=frame.shape[0]

print("\n\n============\n Your resolution is "+str(cols)+"x"+str(rows)+"\n============\n\n\n")

cv2.namedWindow("chroma")


img_back = cv2.imread('esa/marsbig.jpg')
background = img_back[0:rows, 0:cols ]

thresh = 209
blur = 3
final = ""
display=1
kernel = np.ones((5,5),np.uint8)

# create window
#cv2.namedWindow("chroma")#, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("chroma", cv2.WND_PROP_FULLSCREEN, cv2.CV_WINDOW_FULLSCREEN)
ret=True
save_ = True

if save_ and mode=="video":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('_aux'+videonuevo, fourcc, 20.0, (cols, rows))

while(ret):


    key = cv2.waitKey(20)
    #frame = cv2.flip(frame,1,frame)

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
       img_back = cv2.imread('esa/fondo3.png')

    elif k == ord('2'):#11048625: #ord('a'): #1
       img_back = cv2.imread('esa/spacex.jpg')

    elif k == ord('3'):#11048626:#ord('b'): #2
       img_back = cv2.imread('esa/insectos3.png')

    elif k == ord('4'):#11048626:#ord('b'): #2
       img_back = cv2.imread('esa/arbol.png')

    elif k == ord('5'):#11048627:#ord('c'): #3
        img_back = cv2.imread('esa/concavenator3.png')

    elif k == ord('6'):#11048628:#ord('d'): #4
        img_back = cv2.imread('esa/Aragosaurus3.png')

    elif k == ord('7'):#11048629:#ord('e'): #5
        img_back = cv2.imread('esa/titanosaurus3.png')

    elif k == ord('8'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/PIA22107-1024x576.jpg')

    elif k == ord('9'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/curiosity.jpg')

    elif k == ord('0'):#11048630:#ord('f'): #6
        img_back = cv2.imread('esa/mars.jpg')

    if img_back is None:
        print("Image error, check your images!\n\n\n\n")
        break
    if img_back.shape[0]<rows or img_back.shape[1]<cols:
        print("Resolution error, check your images!\n\n\n\n")
        break
    background = img_back[0:rows, 0:cols ]

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

    #Print values
    cv2.putText(dst,"Thresh (+/-): " + str(thresh), (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.5, 0)
    cv2.putText(dst,"Blur (./,): " + str(blur), (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.5, 0)

    #display image
    cv2.imshow('chroma', final)
    if save_ and mode=="video":
        out.write(final)

    if mode=="camera" or mode=="video": # Capture frame-by-frame
        ret, frame = cap.read()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
if save_ and mode=="video":
    out.release()

    video = mp.VideoFileClip('_aux'+videonuevo)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(videonuevo, codec='libx264', audio_codec="aac")
    os.remove('_aux'+videonuevo)

# Original from https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
