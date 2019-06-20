
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from xml.dom import minidom


imagefolder='images'
#Reads items
mydoc = minidom.parse('items.xml')
items = mydoc.getElementsByTagName('marker')
MarkersSize = mydoc.getElementsByTagName('markerssize')

markerssize = 0.10 # default
if MarkersSize!=[]:
    if 'size' in MarkersSize[0].attributes:
        markerssize=float(MarkersSize[0].attributes['size'].value)


class newmarker_:
    file=-1
    markerid=""
    scale=1
    color=""

#Creates dictionary of markers (from items.xml)
dicts = []
for elem in items:
    d=newmarker_()
    for a in elem.attributes.values():
        setattr(d,a.name,a.value)
    dicts.append(d)

def getRepl(markerid__):
    newimage=-1
    scale=1
    color=""
    for elem in dicts:
        if getattr(elem,'markerid')==markerid__ or getattr(elem,'markerid')==str(markerid__):
            newimage=getattr(elem,'file')
            scale=getattr(elem,'scale')
            if newimage=="BGR":
                color=getattr(elem,'color')
            #print(int(scale))
            if scale=="":
                scale=1
    newmarkel = newmarker_
    newmarkel.file=newimage
    newmarkel.scale=int(scale)
    newmarkel.markerid=markerid__
    newmarkel.color=color
    #newmarkel.append(newimage)
    #newmarkel.append(int(scale))
    return newmarkel

def draw(img, corners, imgpts):
     #round(corner[0])
     #round(corner[1])

     corner_ = tuple(corners[0].ravel())
     corner = (int(corner_[0]), int(corner_[1]))


     img0_=tuple(imgpts[0].ravel())
     img0=(int(img0_[0]), int(img0_[1]))


     img1_=tuple(imgpts[1].ravel())
     img1=(int(img1_[0]), int(img1_[1]))

     img2_=tuple(imgpts[2].ravel())
     img2=(int(img2_[0]), int(img2_[1]))

     '''
     img = cv2.line(img, corner, img0, (255,0,0), 5)

     img = cv2.line(img, corner, img1, (0,255,0), 5)

     img = cv2.line(img, corner, img2, (0,0,255), 5)
     '''
     return img


cap = cv2.VideoCapture(1)

####---------------------- CALIBRATION ---------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard of size (7 x 6) is used
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

'''
# iterating through all calibration images
# in the folder
images = glob.glob('calib_images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chess board (calibration pattern) corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

    # if calibration pattern is found, add object points,
    # image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine the corners of the detected corners
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
'''

###------------------ ARUCO TRACKER ---------------------------
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
while (True):
    ret, frame = cap.read()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, markerssize, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error


        length=0.1
        thickness=10
        thicknessSL=100

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            #print (ids[i][0])

            markerelem = getRepl(ids[i][0])
            newfilR = imagefolder+'/'+markerelem.file


            axis = np.float32([[0,0,0],[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            imgpts, jac= cv2.projectPoints(axis, rvec[i], tvec[i], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1,2)

            if newfilR=="BGR":
                color=markerelem.color
                if color=="red":
                    color=(0, 0, 255)
                elif color=="green":
                    color=(0, 255, 0)
                elif color=="blue":
                    color=(255, 0, 0)
                else:
                    color=color[1:len(color)-1]
                    color=tuple(map(int, color.split(', ')))
                length=1
                axis = np.float32([[0,0,0],[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
                imgpts, jac= cv2.projectPoints(axis, rvec[i], tvec[i], mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2)

                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), color, thicknessSL);
                #cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), thicknessSL);
                #cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 0, 0), thicknessSL);
            elif newfilR!=-1:

                axis = np.float32([[0,0,0],[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
                imgpts, jac= cv2.projectPoints(axis, rvec[i], tvec[i], mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2)

                scale=markerelem.scale#markerelem[1]
                if scale=="":
                    scale=1

                #im_src = cv2.imread(newfilR);
                imreal = cv2.imread(newfilR, cv2.IMREAD_UNCHANGED);
                im_src=imreal[:,:,:3]

                transparentbg=False

                if imreal.shape[2]==4:
                    transparentbg=True
                    # Save the transparency channel alpha
                    *_, alpha = cv2.split(imreal)
                    gray_layer = cv2.cvtColor(imreal, cv2.COLOR_BGR2GRAY)
                    gray_layer[np.where((gray_layer>=0).all(axis=1))] = [255]
                    alpha=alpha/255
                    im_src[:,:,1]=im_src[:,:,1]*alpha
                    im_src[:,:,2]=im_src[:,:,2]*alpha
                    im_src[:,:,0]=im_src[:,:,0]*alpha

                    invertedmask=1-alpha


                #M, status=cv2.findHomography(axis, imgpts)



                corner_ = tuple(corners[0].ravel().reshape(-1,2))


                im_dst = frame
                size = im_src.shape

                # Create a vector of source points.

                pts_src = np.array(
                                   [
                                    [0,0],
                                    [int(size[1]), 0],
                                    [int(size[1]), int(size[0])],
                                    [0, int(size[0]) ]
                                    ],dtype=float
                                   )/scale;
                '''
                pts_src = np.array(
                                   [
                                    [-int(size[1]), -int(size[2])],
                                    [-int(size[1]), int(size[2])],
                                    [int(size[1]), int(size[2])],
                                    [int(size[1]), -int(size[2])]
                                    ],dtype=float
                                   )/1;

                '''


                midborder=((corner_[0]+corner_[3])/2)
                #offset=midborder-imgpts[0]
                offset=corner_[0]-imgpts[0]

                corner_2=list(corner_)
                for ni in range(0, 4):
                    corner_2[ni]=(corner_[ni]+5*offset)

                corner_=corner_2


                pts_dst = np.array([corner_[0]-400-int(2*size[1])/scale,corner_[1]-400-int(2*size[1])/scale,corner_[2]-400-int(2*size[1])/scale,corner_[3]-400-int(2*size[1])/scale])#get_four_points(im_dst)

                pts_dst = np.array([corner_[0],corner_[1],corner_[2],corner_[3]])#get_four_points(im_dst)

                mSize=markerssize*3



                objectPoints = np.float32([[0,0,0], [mSize, 0,0], [0,mSize,0], [0,0,mSize]]).reshape(-1,3)

                imgpts2, jac2= cv2.projectPoints(objectPoints, rvec[i], tvec[i], mtx, dist)
                imgpts2 = np.int32(imgpts2).reshape(-1,2)



                h, status = cv2.findHomography(pts_src, pts_dst);
                im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0])) #image src over black

                if not transparentbg:

                    #cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);
                    invertedmask=np.zeros((im_src.shape[0],im_src.shape[1]), np.float32)

                im_tempmask = cv2.warpPerspective(invertedmask, h, (im_dst.shape[1],im_dst.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1)) #image src over white
                im_tempmask=im_tempmask/1

                im_dst[:,:,0]=frame[:,:,0]*im_tempmask
                im_dst[:,:,1]=frame[:,:,1]*im_tempmask
                im_dst[:,:,2]=frame[:,:,2]*im_tempmask



                im_dst=im_dst+im_temp
                frame=im_dst


            else:
                aruco.drawDetectedMarkers(frame, corners)
                #aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)


                #cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], length, thickness)

                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), thickness);
                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), thickness);
                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), thickness);


                '''
                #Deprecated: drawFrameAxes
                cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], length, thickness);
                '''
            cv2.imshow('frame',frame)
        # draw a square around the markers



        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# References
# 1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
# 2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
# 3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
