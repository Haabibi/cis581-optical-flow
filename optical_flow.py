import numpy as np
import cv2
from skimage import transform as tf
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
import math
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
    
    mask[int(y1):int(y2), int(x1):int(x2)] = 255

    features = cv2.goodFeaturesToTrack(img,30,0.01,10, mask=mask)
    corners=np.int32(features)
    numOfFeatures=features.shape[0]
    return numOfFeatures,features


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
    winsize=15
    s=(winsize+1)//2
    x=np.ndarray.item(feature[:,0])
    y=np.ndarray.item(feature[:,1])
    
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    
    x_1=np.linspace(win_l,win_r,winsize) #decimal coord patch image
    y_1=np.linspace(win_t,win_b,winsize)
    xx,yy=np.meshgrid(x_1,y_1)
    img1_window=interp2(img1,xx,yy)
    img2_window=interp2(img2,xx,yy)
    dx_sum=0
    dy_sum=0
    Jx,Jy=calcJxJy(Ix,Iy,xx,yy)
    for i in range(10):
        dx,dy=optical_flow(img1_window,img2_window,Jx,Jy)
        dx_sum+=dx
        dy_sum+=dy
        img2_shift=get_new_img(img2,dx_sum,dy_sum)
        img2_window=interp2(img2_shift,xx,yy)

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
    num_f,FList=extractFeaturefromFeatures(features)
    
    Ix,Iy=findGradient(img2)
    new_features =[]
    
    for idx in range(num_f):
        if FList[idx,0]==0 and FList[idx,1]==0:
            new_features.append(features[idx])
        else:
            new_features.append(estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2))
    return np.array(new_features)


def applyGeometricTransformation(features, new_features, bbox,H,W):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (N, F, 2)
        new_features: Coordinates of all feature points in second frame, (N, F, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    dist_thresh=10
    newFListNum,new_FList=extractFeaturefromFeatures(new_features)
    FListNum,FList=extractFeaturefromFeatures(features)
    
    new_nonZeroFListNum,new_nonZeroFList=extractNonZeroFeature(new_FList)
    nonZeroFListNum,nonZeroFList=extractNonZeroFeature(FList)
    
    tmp_bbox = np.reshape(bbox,(2,-1))
    idx_range=new_nonZeroFListNum
    transform = SimilarityTransform()
    transformation=transform.estimate(nonZeroFList,new_nonZeroFList)
    
    if transformation:
        homoMatrix=transform.params
        transformed_features = matrix_transform(nonZeroFList,homoMatrix)
    for idx in range(idx_range):
        transformed_point=transformed_features[idx]
        new_point=new_nonZeroFList[idx]
        dist_btw_points=math.sqrt((transformed_point[0]-new_point[0])**2 + (transformed_point[1]-new_point[1])**2)
        if dist_btw_points>dist_thresh:
            new_nonZeroFList[idx,:]=np.array([0,0])


    if transformation:
        new_tmp_bbox=matrix_transform(tmp_bbox,homoMatrix)
        tmp_bbox=new_tmp_bbox

    if tmp_bbox[1][0] > W or tmp_bbox[1][1] > H:
        tmp_bbox[1][0]=W
        tmp_bbox[1][1]=H

    for idx in range(new_nonZeroFListNum):
        new_tmp_bbox_x1=tmp_bbox[0][0]
        new_tmp_bbox_y1=tmp_bbox[0][1]
        new_tmp_bbox_x2=tmp_bbox[1][0]
        new_tmp_bbox_y2=tmp_bbox[1][1]
        if new_nonZeroFList[idx][0] < new_tmp_bbox_x1 or new_nonZeroFList[idx][1]<new_tmp_bbox_y1 or new_nonZeroFList[idx][0]>new_tmp_bbox_x2 or new_nonZeroFList[idx][1]>new_tmp_bbox_y2:
            new_nonZeroFList[idx]=[0,0]

    new_bbox=new_tmp_bbox.reshape(1,2,2)
    features_fillzeros=np.zeros((FListNum,2))

    features_fillzeros[:new_nonZeroFListNum,:]=new_nonZeroFList

    features_fillzeros=features_fillzeros.reshape(FListNum,1,-1)

    return features_fillzeros, new_bbox


