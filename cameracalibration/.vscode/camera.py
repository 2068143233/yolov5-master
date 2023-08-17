import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
 
# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Detect corners on a chessboard
 
filename = 'p1.jpg'
 
# Chessboard dimensions
number_of_squares_X = 7 # Number of chessboard squares along the x-axis
number_of_squares_Y = 9  # Number of chessboard squares along the y-axis
nX = number_of_squares_X - 1 # Number of interior corners along x-axis
nY = number_of_squares_Y - 1 # Number of interior corners along y-axis
# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)       
 
# These are the x and y coordinates                                              
object_points_3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) 
objpoints=[]
def main():

    # Load an image
    image = cv2.imread(filename)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image",gray)
    # cv2.waitKey(0)
    # Find the corners on the chessboard
    success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
    # If the corners are found by the algorithm, draw them
    if success == True:
        objpoints.append(object_points_3D)
        # Draw the corners
        cv2.drawChessboardCorners(image, (nY, nX), corners, success)
        # Create the output file name by removing the '.jpg' part
        size = len(filename)
        new_filename = filename[:size - 4]
        new_filename = new_filename + '_drawn_corners.jpg'     
        
        # Save the new image in the working directory
        # cv2.imwrite(new_filename, image)
        # print(corners)
        # Display the image 
        # cv2.imshow("Image", image)
        
        # Display the window until any key is pressed
        # cv2.waitKey(0) 
        
        # Close all windows
        # cv2.destroyAllWindows() 
    # h,w=image.shape[:2]
    print('adjsohfnjk')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, corners, gray.shape[::-1], None, None)
    print(("ret:"), ret)
    print(("mtx:\n"), mtx)  # 内参数矩阵
    print(("dist:\n"), dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print(("rvecs:\n"), rvecs)  # 旋转向量  # 外参数
    print(("tvecs:\n"), tvecs)  # 平移向量  # 外参数
main()