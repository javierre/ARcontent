import cv2

# File storage in OpenCV
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

print("camera_matrix : ", mtx.tolist())
print("dist_matrix : ", dist.tolist())

cv_file.release()
