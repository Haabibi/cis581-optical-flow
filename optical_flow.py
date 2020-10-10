import numpy as np
import cv2
from skimage import transform as tf

from helpers import *


def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (N, F, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
        """
    y1=np.ndarray.item(bbox[:,0,1])
    y2=np.ndarray.item(bbox[:,1,1])
    x1=np.ndarray.item(bbox[:,0,0])
    x2=np.ndarray.item(bbox[:,1,0])
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    features = cv2.goodFeaturesToTrack(img,25,0.01,10, mask=mask)
    print("THIS FEATURES", features)
    corners=np.int32(features)
    img_to_show = img.copy()
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_to_show,(x,y),3,(0,0,255),5)
    
    cv2.imwrite("result.jpg",img_to_show*255)
    return features


def estimateFeatureTranslation(feature, Ix, Iy, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        Ix: Gradient along the x direction, (H,W)
        Iy: Gradient along the y direction, (H,W)
        img1: First image frame, (H,W)
        img2: Second image frame, (H,W)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """
    winsize=30
    s=(winsize+1)//2
    x=np.ndarray.item(feature[:,0])
    y=np.ndarray.item(feature[:,1])
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    win_l = np.round(win_l).astype(int)
    win_r = np.round(win_r).astype(int)
    win_t = np.round(win_t).astype(int)
    win_b = np.round(win_b).astype(int)
    img1_window=img1[win_t:win_b,win_l:win_r].copy()
    #print("THIS IS IMG WINDOW", img1_window)
    img2_window=img2[win_t:win_b,win_l:win_r].copy()
    dx_sum=0
    dy_sum=0
    for i in range(30):
        #print("BEFORE OF: ", img1_window.shape, img2_window.shape)
        #if img2_window.shape != (16, 16):
        #    break
        dx,dy=optical_flow(img1_window,img2_window,5,1)
        dx_sum+=dx
        dy_sum+=dy
        if i == 0:
            print(" INIT DX DY", dx, dy, dx_sum, dy_sum)
        if i == 29:
            print(" FINAL DX DY", dx, dy, dx_sum, dy_sum)
        img2_shift=get_new_img(img2,dx_sum,dy_sum)
        img2_window=img2_shift[win_t:win_b,win_l:win_r].copy()

    dx_sum=np.round(dx_sum).astype(int)
    dy_sum=np.round(dy_sum).astype(int)
    new_feature = feature + np.array((dx_sum, dy_sum))
    return new_feature


def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (N, F, 2)
        img1: First image frame, (H,W)
        img2: Second image frame, (H,W)
    Output:
        new_features: Coordinates of all feature points in second frame, (N, F, 2)
    """
    #print("THIS IS FEATURES", features)
    interest_num=features.shape[1]
    num_f=features.shape[0]
    
    Ix,Iy=findGradient(img2)
    new_features =[]
    for idx in range(num_f):
        new_features.append(estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2))
    #print(new_features)
    return np.array(new_features)


def applyGeometricTransformation(features, new_features, bbox):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    features, bbox = None, None
    return features, bbox


