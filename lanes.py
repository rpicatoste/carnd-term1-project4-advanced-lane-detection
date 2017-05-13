
#%%

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import camera_calibration as cc
import color_and_gradient as cg


def test_functions():
    
    image_file = r'.\CarND-Advanced-Lane-Lines\test_images\test2.jpg'
    image = cv2.imread( image_file )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    
    # undistort and unwarp
    warped = cc.undistort_and_warp(image)
      
    warped_binary = cg.pipeline_filter_to_binary(warped, s_thresh = (150, 255), l_thresh = (210, 255), plot_layers = False )
    y, left_x, left_y, left_fit, right_fit = fit_lane_to_binary_image( warped_binary )
    left_rad, right_rad = get_curvature(y, left_fit, right_fit)
   

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))     
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title(image_file)
    ax2.imshow(warped_binary, cmap = 'gray')
    ax2.plot(left_x, y, color='yellow')
    ax2.plot(left_y, y, color='yellow')
    ax2.set_xlim(0              , image.shape[1])
    ax2.set_ylim(image.shape[0] , 0)
    ax2.set_title('1st pass. Curvature: left {: 4} m, right {: 4} m.'.format( int(left_rad), int(right_rad) ) )
    
    y, left_x, left_y, left_fit, right_fit = fit_lane_to_binary_image_from_previous_fit(warped_binary, left_fit, right_fit )
    left_rad, right_rad = get_curvature(y, left_fit, right_fit)
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))     
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title(image_file)
    ax2.imshow(warped_binary, cmap = 'gray')
    ax2.plot(left_x, y, color='yellow')
    ax2.plot(left_y, y, color='yellow')
    ax2.set_xlim(0              , image.shape[1])
    ax2.set_ylim(image.shape[0] , 0)
    ax2.set_title('2nd pass. Curvature: left {: 4} m, right {: 4} m.'.format( int(left_rad), int(right_rad) ) )
    
    


# Function to fit a second order polynomial to the 2 lanes from a bird-view 
# image, filtered with the color pipeline.
def fit_lane_to_binary_image( binary_warped, plot_figure = False ):
        
    binary_warped = binary_warped.astype(np.uint8)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Take a histogram of the bottom half of the image
    histogram = np.sum( binary_warped[binary_warped.shape[0]//2:,:], axis = 0 )
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
 
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] -  window    * window_height
        win_xleft_low  = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    
    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print('Empty vector arrived!!',lefty, leftx, righty, rightx )
        return None, None, None, None, None

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx =   left_fit[0]*ploty**2 +  left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if plot_figure:
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return ploty, left_fitx, right_fitx, left_fit, right_fit
    

# Function to fit a second order polynomial to the 2 lanes from a bird-view 
# image, filtered with the color pipeline, taking into account previous 
# fittings from previous frames.
def fit_lane_to_binary_image_from_previous_fit(binary_warped, left_fit, right_fit, plot_figure = False ):
    
    binary_warped = binary_warped.astype(np.uint8)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = (  (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)) ) 
    
    right_lane_inds =(  (nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                      & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)) )  
        
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print('Empty vector arrived!!')
        return None, None, None, None, None
        
        
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx  =  left_fit[0]*ploty**2 +  left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Draw the lane onto the warped blank image
    if plot_figure:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[ left_lane_inds], nonzerox[ left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        plt.figure()
        plt.imshow(result)
        plt.plot( left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return ploty, left_fitx, right_fitx, left_fit, right_fit


# Get the curvature radius from the coefficients obtained from fitting the lane
# image to second order polynomials.
def get_curvature(ploty, left_fit, right_fit, plot_figure = False):
   
    left_fitx  =  left_fit[0]*ploty**2 +  left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)    

    # Define conversions in x and y from pixels space to meters
    # They are approximately as the values suggested in the lesson (30 meters 
    # of lane seen in the bird view, even though this is very difficult to 
    # judge), and 3.7 meters wide (this is regulation, so it is more reliable).
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr  = np.polyfit(ploty * ym_per_pix,  left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad_m  = ((1 + (2* left_fit_cr[0]*y_eval*ym_per_pix +  left_fit_cr[1])**2)**1.5) / np.absolute(2* left_fit_cr[0])
    right_curverad_m = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    if plot_figure:
        # Plot up the fake data
        plt.figure()
        plt.plot(left_fitx, ploty, color='red', linewidth=3)
        plt.plot(right_fitx, ploty, color='blue', linewidth=3)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.title('Curvature: left {: 4} m, right {: 4} m.'.format( int(left_curverad_m), int(right_curverad_m) ) )
        
    return left_curverad_m, right_curverad_m
    


def fill_found_lanes_in_original( image, y, left_fit, right_fit):
    
    left_x =  left_fit[0]*y**2 +  left_fit[1]*y +  left_fit[2]
    left_y = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_x, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([left_y, y])))])
    poly_points = np.hstack((pts_left, pts_right))
    
    warped_grays = np.zeros_like( image )
    #warped_grays = np.dstack((black_binary, black_binary, black_binary))
    cv2.fillPoly(warped_grays, np.int_([poly_points]), (0, 200, 0))
        
    unwarped_grays_fill = cc.unwarp( warped_grays.astype(np.int16) ).astype(np.uint8)
    result = cv2.addWeighted(image, 1, unwarped_grays_fill, 0.4, 0)
    
    return result


if __name__ == '__main__':
    test_functions()