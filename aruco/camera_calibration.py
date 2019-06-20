import numpy as np
import cv2
import glob


# checkerboard Dimensions
cbrow = 9
cbcol = 7

# True for offline processing
# False for online usb-camera processing (until you press ESC)
offlineprocess=True

WAIT_TIME = 10
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow*cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



def findCorners(gray_, img_, fname_):
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_, (cbcol,cbrow),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray_,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img_ = cv2.drawChessboardCorners(img_, (cbcol,cbrow), corners2,ret)
        cv2.imshow('img',img_)
        cornersfile=fname_.replace('calib_images/','calib_images/corners/')
        cv2.imwrite(cornersfile, img_)
        cv2.waitKey(WAIT_TIME)





if offlineprocess:
    '''
    Previously recorded images
    '''
    images = glob.glob('calib_images/frame*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        findCorners(gray, img, fname)
else:
    '''
    Online processing
    '''
    cap = cv2.VideoCapture(1)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(33)
        if k == 27: #ESC
            break
        fname='online'+cap.get(cv2.CAP_PROP_FRAME_COUNT)
        findCorners(gray, frame, fname)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)
# note you *release* you don't close() a FileStorage object
cv_file.release()
