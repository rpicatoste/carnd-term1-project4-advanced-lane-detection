#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Parameters to select ROI as fraction of the image.
imshape_used = (720, 1280, 3)

# These are sets of points, where for a certain height of the points used for 
# transform, there is a corresponding lateral distance (so the points will be 
# in the lanes for the sample images with straight lanes).
sets_of_corresponding_points = [ [440, 608] ,
                                 [500, 520] ]
set_number_to_use = 0
top_limit        = sets_of_corresponding_points[set_number_to_use][0]
top_lateral_diff = sets_of_corresponding_points[set_number_to_use][1]

bottom_limit = 720

pts_src = np.array([ [200,                                  bottom_limit],
                     [top_lateral_diff,                     top_limit], 
                     [imshape_used[1]-top_lateral_diff,     top_limit], 
                     [1080,                                 bottom_limit]], np.float32)

distance_to_sides = 300        
pts_dst = np.array([[distance_to_sides,                   720],
                    [distance_to_sides,                     0],
                    [imshape_used[1] - distance_to_sides,   0],
                    [imshape_used[1] - distance_to_sides, 720] ],np.float32)

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def sample_to_find_transformation_points():
    
    # Get camera matrices
    camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
    mtx = camera_dict["mtx"] 
    dist = camera_dict["dist"] 

    image_files = [r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg',
                  r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg']
    
    for image_file in image_files:
        image = cv2.imread( image_file )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
        
        imshape = imshape_used
            
        # undistort and unwarp
        image = cv2.undistort(image, mtx, dist, None, mtx)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        Minv = cv2.getPerspectiveTransform(pts_dst, pts_src)
        warped = cv2.warpPerspective(image, M,  (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        
        image = cv2.circle(image, tuple(pts_src[0]), 10, color = [255,0,0], thickness = -1)
        image = cv2.circle(image, tuple(pts_src[1]), 10, color = [0,0,255], thickness = -1)
        image = cv2.circle(image, tuple(pts_src[2]), 10, color = [0,255,0], thickness = -1)
        image = cv2.circle(image, tuple(pts_src[3]), 10, color = [255,255,0], thickness = -1)
        
        warped = cv2.circle(warped, tuple(pts_dst[0]), 10, color = [255,0,0], thickness = -1)
        warped = cv2.circle(warped, tuple(pts_dst[1]), 10, color = [0,0,255], thickness = -1)
        warped = cv2.circle(warped, tuple(pts_dst[2]), 10, color = [0,255,0], thickness = -1)
        warped = cv2.circle(warped, tuple(pts_dst[3]), 10, color = [255,255,0], thickness = -1)
        
        unwarped = cv2.warpPerspective(warped, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        
        
        f, (ax1, ax2) = plt.subplots(2,2, figsize=(12, 8))
        f.tight_layout()
        ax1[0].imshow(image)
        ax1[0].set_title('Original Image')
        ax1[1].imshow(warped)
        ax1[1].set_title('Warped')
        ax2[1].imshow(unwarped)
        ax2[1].set_title('Un-Warped')
       
def get_camera_matrices_and_perspective_transform():
    camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
    mtx = camera_dict["mtx"] 
    dist = camera_dict["dist"] 
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    return mtx, dist, M
    
    
def undistort_and_warp( image, mtx = None, dist = None, M = None):    
    # Get camera matrices
    if (mtx is None) or (dist is None) or (M is None):
        mtx, dist, M = get_camera_matrices_and_perspective_transform()
    
    image = cv2.undistort(image, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(image, M,  (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped        
    

def get_camera_inverse_perspective_transform():
    
    Minv = cv2.getPerspectiveTransform(pts_dst, pts_src)
    
    return Minv

def unwarp( warped, Minv = None):    
    # Get camera matrices
    if (Minv is None):
        Minv = get_camera_inverse_perspective_transform()

    unwarped = cv2.warpPerspective(warped, Minv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
    
    return unwarped        
    

def calibrate_camera():
    
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('CarND-Advanced-Lane-Lines\camera_cal\calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    # Test undistortion on an image. From the camera calibration folder, the one 
    # that looks more distorted is the 1.
    img = cv2.imread('CarND-Advanced-Lane-Lines\camera_cal\calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('camera_calibration_test_undist.jpg',dst)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    camera_dict = {}
    camera_dict["mtx"] = mtx
    camera_dict["dist"] = dist
    pickle.dump( camera_dict, open( "camera_calibration.p", "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(2, figsize=(10,13))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=20)
    


    # Undistort and transform
    camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
    
    mtx = camera_dict["mtx"] 
    dist = camera_dict["dist"] 
    
    fname = '.\CarND-Advanced-Lane-Lines\camera_cal\calibration3.jpg'
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    
    
    top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 7))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)




def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    img_dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    img_size = (img.shape[1], img.shape[0])
    
    # 2) Convert to grayscale
    gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
    # 4) If corners found: 
    if ret:
        # a) draw corners
        cv2.drawChessboardCorners(img_dst, (nx,ny), corners, ret)
        
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
             #Note: you could pick any four of the detected corners 
             # as long as those four corners define a rectangle
             #One especially smart way to do this would be to use four well-chosen
             # corners that were automatically detected during the undistortion steps
             #We recommend using the automatic detection of corners in your code
        
        # Draw and display the corners
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
#        cv2.imshow('img', img)
        
        pts_src = corners.squeeze()[ [0,nx-1,-(nx),-1], ]
        
        plt.figure( figsize=(10,5) )
        plt.imshow(img_dst)
        plt.plot(pts_src[0][0],pts_src[0][1], '.r')
        plt.plot(pts_src[1][0],pts_src[1][1], '.b')
        plt.plot(pts_src[2][0],pts_src[2][1], '.y')
        plt.plot(pts_src[3][0],pts_src[3][1], '.g')


        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        x_in = 100
        x_fin = img_size[1] - 100
        y_in = 100
        y_fin = img_size[0] - 100
        global pts_dst, pts_src
        pts_dst = np.float32([[y_in,  x_in],
                              [y_fin, x_in],
                              [y_in,  x_fin],
                              [y_fin, x_fin] ])
    
        plt.plot(pts_dst[0][0],pts_dst[0][1], '+r')
        plt.plot(pts_dst[1][0],pts_dst[1][1], '+b')
        plt.plot(pts_dst[2][0],pts_dst[2][1], '+y')
        plt.plot(pts_dst[3][0],pts_dst[3][1], '+g')

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img_dst, M, img_size, flags=cv2.INTER_LINEAR)
        
    return warped, M

if __name__ == '__main__':
#    calibrate_camera()

    sample_to_find_transformation_points()