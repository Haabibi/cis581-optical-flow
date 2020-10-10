import numpy as np
import cv2
from skimage import transform as tf
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
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

    features = cv2.goodFeaturesToTrack(img,5,0.01,10, mask=mask)
    corners=np.int32(features)img_to_show = img.copy()
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
    It=img2-img1
    nr=np.arange(img1.shape[0])
    nc=np.arange(img1.shape[1])

    winsize=15
    s=(winsize+1)//2
    x=np.ndarray.item(feature[:,0])
    y=np.ndarray.item(feature[:,1])
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    
    win_l = int(win_l)
    win_r = int(win_r)
    win_t = int(win_t)
    win_b = int(win_b)
    img1_window=img1[win_t:win_b,win_l:win_r]
    img2_window=img2[win_t:win_b,win_l:win_r]
    dx_sum=0
    dy_sum=0
    for i in range(30):
        dx,dy=optical_flow(img1_window,img2_window,5,1)
        dx_sum+=dx
        dy_sum+=dy
        img2_shift=get_new_img(img2,dx_sum,dy_sum)
        img2_shift_to_show = img2_shift.copy()
        cv2.imwrite("shift_{}.jpg".format(i), img2_shift_to_show*255)
        img2_window=img2_shift[win_t:win_b,win_l:win_r]

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
    num_f=features.shape[0]
    Ix,Iy=findGradient(img2)
    new_features =[]
    for idx in range(num_f):
        new_features.append(estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2))
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
    num_features = features.shape[0]
    tmp_features = features.reshape((num_features, -1))
    tmp_new_features = new_features.reshape((num_features,-1))
    tmp_bbox = bbox.reshape(2,-1)

    transform = SimilarityTransform()
    transformation = transform.estimate(tmp_features, tmp_new_features)
    
    if transformation:
        homoMatrix = transform.params
        new_bbox = matrix_transform(tmp_bbox, homoMatrix)
    else:
        new_bbox = tmp_bbox

    features, bbox = features, new_bbox


    return features, bbox


