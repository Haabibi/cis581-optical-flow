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
    crop_img=np.copy(img[y1:y2,x1:x2])
    
    features = cv2.goodFeaturesToTrack(crop_img,7,0.01,30)
    corners=np.int32(features)
    map_to_original = np.array([[[x1, y1] for _ in range(len(corners))]]).reshape(corners.shape) + corners
    for i in map_to_original:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,(0,0,255),3)
    
    cv2.imwrite("result.jpg",img*255)
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
    It=img2-img1
    nr=np.arange(img1.shape[0])
    nc=np.arange(img1.shape[1])
#    i_It=interp2(It,nc,nr)
#    i_Ix=interp2(Ix,nc,nr)
#    i_Iy=interp2(Iy,nc,nr)

    winsize=15
    s=(winsize+1)//2
    x=np.ndarray.item(feature[:,0])
    y=np.ndarray.item(feature[:,1])
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    print("x,y",x,y)
    x=int(x)
    y=int(y)
    img1_window=select_win([img1],slice(y-s,y+s),slice(x-s,x+s))
    img2_window=select_win([img2],slice(y-s,y+s),slice(x-s,x+s))
    dx_sum=0
    dy_sum=0
    for i in range(30):
        dx,dy=optical_flow(img1_window,img2_window,winsize,1)
        print("DX and DY",dx,dy)
        dx_sum+=dx
        dy_sum+=dy
        img1_shift=get_new_img(img1,dx_sum,dy_sum)
        img1_window=select_win([img1_shift],slice(y-s,y+s),slice(x-s,x+s))

    
    new_feature = img1_window
    print("THIS IS RES", res)
    print("THIS IS NEW FEATURE", new_feature)
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
    print("THIS IS FEATURES", features)
    interest_num=features.shape[1]
    num_f=features.shape[0]
    
    Ix,Iy=findGradient(img2)
    new_features =[]
    for idx in range(num_f):
        new_features.append(estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2))
    #print(new_features)
    return new_features


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


