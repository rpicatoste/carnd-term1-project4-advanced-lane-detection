
#%% Applying sobel

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle



def run_examples():
    
    # Read in an image and grayscale it
    image = mpimg.imread(r'.\course\test6.jpg') 
#    image = mpimg.imread('course\signs_vehicles_xygrad.png') # imagen con de 0 a 1 con mpimg
    
    # Directional gradient
    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x', thres=(20,100) )
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # Magnitude of the gradient
    # Run the function
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    # Plot the result
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Thresholded Magnitude')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


    # Direction of the Gradient    
    # Run the function
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    # Plot the result
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(dir_binary, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Combine
    # Choose a Sobel kernel size
    ksize = 15
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', thres=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', thres=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    #final_val = gradx * grady * mag_binary * dir_binary
    final_val = np.zeros_like(dir_binary)
    final_val[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # Plot the result
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(final_val, cmap='gray')
    ax2.set_title('My own combination.')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Color
    # Read in an image, you can also try test1.jpg or test4.jpg
    image = mpimg.imread(r'.\course\test6.jpg') 
  
    hls_binary = hls_select(image, thresh=(90, 255), layer = 'S', plot_layers = False)
#    hls_binary = hls_select(image, thresh=(90, 255))
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(hls_binary, cmap='gray')
    ax2.set_title('Thresholded S')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    
    # Color and gradient pipeline
    #image = mpimg.imread('bridge_shadow.jpg')
#    image = mpimg.imread('course\signs_vehicles_xygrad.png')
    image = mpimg.imread(r'.\course\test4.jpg')    
    result = pipeline_filter_to_binary(image)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 7))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    
    ax2.imshow(result)
    ax2.set_title('Pipeline Result')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    

#%% Functions
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
#def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
def abs_sobel_thresh(img, orient='x', thres = (9,255) ):
#    
    thresh_min = thres[0]
    thresh_max = thres[1]
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
        
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))    
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#    plt.imshow(sxbinary, cmap='gray')
            
    # 6) Return this mask as your binary_output image
    return sbinary
    
    
# Magnitude of the gradient

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
        
    # 3) Calculate the magnitude 
    mag = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255*mag/np.max(mag))    

    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_mag)
    sbinary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
            
    # 6) Return this mask as your binary_output image
    return sbinary       


# Direction of the Gradient

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
        
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_arg = np.arctan2(abs_sobely, abs_sobelx)    
    
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(grad_arg)
    sbinary[(grad_arg >= thresh[0]) & (grad_arg <= thresh[1])] = 1
            
    # 6) Return this mask as your binary_output image
    return sbinary    
    


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255), layer = 'S', plot_layers = False):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    # 2) Apply a threshold to the S channel
    sbinary = np.zeros_like(S)
    if layer.upper() == 'L':
        sbinary[ (L > thresh[0]) & (L <= thresh[1]) ] = 1
    elif layer.upper() == 'H':
        sbinary[ (H > thresh[0]) & (H <= thresh[1]) ] = 1
    else:
        sbinary[ (S > thresh[0]) & (S <= thresh[1]) ] = 1
        
    # 3) Return a binary image of threshold result
    if plot_layers:
        
        f, (ax1, ax2) = plt.subplots(2,2, figsize=(11, 8))
        f.tight_layout()
        ax1[0].imshow(img)
        ax1[0].set_title('Original Image')
        ax1[1].imshow(S, cmap='gray')
        ax1[1].set_title('S Layer, max:'+ str(np.max(S)) + ', min:' + str(np.min(S)) )
        ax2[0].imshow(H, cmap='gray')
        ax2[0].set_title('H Layer, max:'+ str(np.max(H)) + ', min:' + str(np.min(H)) )
        ax2[1].imshow(L, cmap='gray')
        ax2[1].set_title('L Layer, max:'+ str(np.max(L)) + ', min:' + str(np.min(L)) )
#        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        pass
    
    return sbinary
  


# Color and gradient pipeline

# Edit this function to create your own pipeline.
def pipeline_filter_to_binary(img, s_thresh = (120, 255), l_thresh = (210, 255) , plot_layers = False ):
    img = np.copy(img)
    
    binary_S = hls_select(img, s_thresh, 'S', plot_layers )
    binary_L = hls_select(img, l_thresh, 'L', False)
    binary = np.logical_or( binary_S, binary_L )

    return binary



if __name__ == '__main__':
    run_examples()