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
    # y1=np.ndarray.item(bbox[:,0,1])
    # y2=np.ndarray.item(bbox[:,1,1])
    # x1=np.ndarray.item(bbox[:,0,0])
    # x2=np.ndarray.item(bbox[:,1,0])
    print("THIS IS BBOX SHPAE", bbox.shape)
    y1 = int(bbox[0,1])
    y2 = int(bbox[1,1])
    x1 = int(bbox[0,0])
    x2 = int(bbox[1,0])
    print("THIS IS IN GET FEATURES", y1, y2, x1, x2)
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    features = cv2.goodFeaturesToTrack(img,30,0.01,10, mask=mask)
    features_squeeze = features.reshape(-1, 2)
    corners=np.int32(features_squeeze)
    img_to_show = img.copy()
    
    #for (x,y) in corners:
        #x,y = i.ravel()
    #    cv2.circle(img_to_show,(x,y),3,(0,0,255),3)
    
    #cv2.imwrite("result.jpg",img_to_show*255)
    print("THIS AT GET FEAUTRES", features_squeeze)
    return features_squeeze


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
    #print("THIS IS FEATURES in ESTIM", feature.shape)
    #x=np.ndarray.item(feature[:,0])
    #y=np.ndarray.item(feature[:,1])
    x = feature[0]
    y = feature[1]
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    
    x_1=np.linspace(win_l,win_r,winsize) #decimal coord patch image
    y_1=np.linspace(win_t,win_b,winsize)
    xx,yy=np.meshgrid(x_1,y_1)
    
    img1_window=interp2(img1,xx,yy)
    img2_window=interp2(img2,xx,yy)
    
    dx_sum=0
    dy_sum=0
    for i in range(10):
        dx,dy=optical_flow(img1_window,img2_window,Ix,Iy, xx,yy)
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
    num_f=features.shape[0]
    print("IN ESTIMATE ALL TRANS", features.shape, num_f)
    Ix,Iy=findGradient(img2)
    new_features =[]
    for idx in range(num_f):
        new_features.append(estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2))
    return np.array(new_features)


def applyGeometricTransformation(frame, features, new_features, bbox):
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
    print("THIS IS FEATURES", features)
   # print("AT APPLY GEOMETRIC", new_features.shape)
    dist_thresh = 20
    # filter out 0,0s 
    #print("THIS IS SIZE NEW_ FEAT", new_features.shape) #30,2
    non_zero_mask = np.logical_or(features[:,0] > 1, features[ :,1]>1)
    new_features = new_features[non_zero_mask]
    features = features[non_zero_mask]
    print("THIS IS NEW _FEAUTES", new_features)
    num_features = features.shape[0]
    print("THIS IS NUM_ FEATURES", num_features)

    if num_features < 30*0.4:
        get_new_feat = getFeatures(frame, bbox)
        print("!!!!!!BEFORE NEW FEAT", new_features.shape, get_new_feat.shape)
        #new_features = np.concatenate((new_features, get_new_feat), axis=0)
        #features = np.concatenate((features, get_new_feat), axis=0)
        print("THIS IS NEW FEATURES!!!!!!!", new_features.shape)
        num_features= features.shape[0]

    transform = SimilarityTransform()
    transformation = transform.estimate(features, new_features)
    if transformation:
        homoMatrix = transform.params
        transformed_features = matrix_transform(features, homoMatrix)
    else:
        print("NO TRANSFORMED ON FEATURES\n")
    for idx in range(num_features):
        transformed_point = transformed_features[idx]
        new_point = new_features[idx]
        dist_btw_points = math.sqrt((transformed_point[0]- new_point[0])**2 + (transformed_point[1] - new_point[1])**2)
        print("THis is DIST POINTS", dist_btw_points)
        if dist_btw_points > dist_thresh:
            new_features[idx,:] = np.array([0, 0])

    if transformation:
        bbox = matrix_transform(bbox, homoMatrix)
    else:
        print("NO TRANSFORMED ON BBOX")
    #print("THIS IS NEW BBOX", bbox)
    #new_bbox = new_bbox.reshape(1, 2, 2)
    #filterout 
    # may want to filter out 
    if idx in range(num_features):
        new_bbox_x1 = bbox[0][0]
        new_bbox_y1 = bbox[0][1]
        new_bbox_x2 = bbox[1][0]
        new_bbox_y2 = bbox[1][1]
        #print("THIS IS NEW BBOX X1 to Y2", new_bbox_x1, new_bbox_x2, new_bbox_y1, new_bbox_y2)
        if new_features[idx][0] < new_bbox_x1 or new_features[idx][1] < new_bbox_y1 or new_features[idx][0] > new_bbox_x2 or new_features[idx][1] > new_bbox_y2:
            new_features[idx] = [0, 0]
   #new_zero_mask = np.logical_or(new_features[:,0] > 1, new_features[ :,1]>1)
    #new_features = new_features[non_zero_mask]
   #new_bbox = new_bbox.reshape(-1, 2, 2)
    #print("RETURNED NEW BBOX", bbox, new_features)
    return new_features, bbox
            
    """
    tmp_features = features.reshape((num_features, -1)) #5,2
    tmp_new_features = new_features.reshape((num_features,-1)) #5,2
    tmp_bbox = bbox.reshape(2,-1)
    dist_thresh = 8
    for idx in range(num_features):
        old_point = tmp_features[idx]
        new_point = tmp_new_features[idx]
        dist_btw_points = math.sqrt((old_point[0] - new_point[0])**2 + (old_point[1]-new_point[1])**2)
        print("DIST BETWEEN POINTS:", dist_btw_points)
        if dist_btw_points > dist_thresh:
            tmp_features[idx,:] = np.array([0, 0])
            tmp_new_features[idx,:] = np.array([0, 0])

    mask=np.logical_or(tmp_features[:,0]>0,tmp_features[:,1]>0)
    tmp_tmp_features = tmp_features[mask,:]
    tmp_tmp_new_features=tmp_new_features[mask,:]

    transform = SimilarityTransform()
    transformation = transform.estimate(tmp_tmp_features, tmp_tmp_new_features)
    
    if transformation:
        homoMatrix = transform.params
        new_bbox = matrix_transform(tmp_bbox, homoMatrix)
    else:
        new_bbox = tmp_bbox
    
    features, bbox = tmp_new_features.reshape(num_features,1,-1), new_bbox.reshape(1,2,2)

    if len(tmp_tmp_new_features) < num_features * 0.6:
        bbox = bbox.reshape(1, 2, 2)
        bbox = bbox.astype(int)
        new_features = getFeatures(frame, bbox)
        features = new_features

    return features, bbox

    """
