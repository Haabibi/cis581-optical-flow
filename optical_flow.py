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
    #print("THIS IS IN GET FEATURES", y1, y2, x1, x2)
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    mask[int(y1):int(y2), int(x1):int(x2)] = 255

    features = cv2.goodFeaturesToTrack(img,30,0.01,10, mask=mask)
    corners=np.int32(features)
    img_to_show = img.copy()
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_to_show,(x,y),3,(0,0,255),3)
    
    cv2.imwrite("result.jpg",img_to_show*255)
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
    #print("ESTIMEATE/FEATURE",feature,feature.shape,feature[:,0],feature[:,1])
    x=np.ndarray.item(feature[:,0])
    y=np.ndarray.item(feature[:,1])
    
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    
    x_1=np.linspace(win_l,win_r,winsize) #decimal coord patch image
    y_1=np.linspace(win_t,win_b,winsize)
    #print("x1, y1",x_1,y_1)
    xx,yy=np.meshgrid(x_1,y_1)
    win_l=int(win_l)
    win_r=int(win_r)
    win_t=int(win_t)
    win_b=int(win_b)
    img1_window=interp2(img1,xx,yy)
    img2_window=interp2(img2,xx,yy)
    #img1_window=img1[win_t:win_b,win_l:win_r]
    #img2_window=img2[win_t:win_b,win_l:win_r]
    dx_sum=0
    dy_sum=0
    for i in range(30):
        dx,dy=optical_flow(img1_window,img2_window,Ix,Iy, xx,yy)
        dx_sum+=dx
        dy_sum+=dy
        img2_shift=get_new_img(img2,dx_sum,dy_sum)
        img2_window=img2_shift[win_t:win_b,win_l:win_r]
    #img2_window=interp2(img2_shift,xx,yy)

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
    #print("ALL",features,features.shape)
    #num_f=features.shape[0]
    num_f,FList=extractFeaturefromFeatures(features)
    
    Ix,Iy=findGradient(img2)
    new_features =[]
    
    for idx in range(num_f):
        if FList[idx,0]==0 and FList[idx,1]==0:
            new_features.append(features[idx])
        else:
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
    #num_features = features.shape[0]
    """
    tmp_features = features.reshape((num_features, -1)) #5,2
    print("shape",tmp_features.shape)
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
                #print("TMP_FEATURES",tmp_features)

                #print("TMP_NEW_FEATURES",tmp_new_features)
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

    features, bbox = tmp_new_features.reshape(1,num_features,-1), new_bbox.reshape(1,2,2)

    if len(tmp_tmp_new_features) < num_features * 0.6:
        bbox = bbox.reshape(1, 2, 2)
        bbox = bbox.astype(int)
        new_features = getFeatures(frame, bbox)
        features = new_features
    """
    dist_thresh=10
#    tmp_new_features=new_features.reshape((num_features,-1))
#    tmp_features=features.reshape((num_features,-1))
#    non_zero_mask=np.logical_or(tmp_features[:,0]>0,tmp_features[:,1]>0)
#    new_features=tmp_new_features[non_zero_mask,:]
#    features = tmp_features[non_zero_mask,:]
    newFListNum,new_FList=extractFeaturefromFeatures(new_features)
    FListNum,FList=extractFeaturefromFeatures(features)
    
    new_nonZeroFListNum,new_nonZeroFList=extractNonZeroFeature(new_FList)
    nonZeroFListNum,nonZeroFList=extractNonZeroFeature(FList)
    
    tmp_bbox = np.reshape(bbox,(2,-1))
    #idx_range=features.shape[0]
    idx_range=new_nonZeroFListNum
    #print("TMP BBOX\n",tmp_bbox,tmp_bbox.shape)
    #print("NEW_FEATURES WITH ZERO",new_FList,new_FList.shape)
    transform = SimilarityTransform()
    #transformation=transform.estimate(features,new_features)
    transformation=transform.estimate(nonZeroFList,new_nonZeroFList)
    
    if transformation:
        homoMatrix=transform.params
        transformed_features = matrix_transform(nonZeroFList,homoMatrix)
    for idx in range(idx_range):
        transformed_point=transformed_features[idx]
        new_point=new_nonZeroFList[idx]
        dist_btw_points=math.sqrt((transformed_point[0]-new_point[0])**2 + (transformed_point[1]-new_point[1])**2)
        #print("DISTANCE",dist_btw_points)
        if dist_btw_points>dist_thresh:
            new_nonZeroFList[idx,:]=np.array([0,0])

#print("NEWFEATURES\n",new_nonZeroFList)



    if transformation:
        new_tmp_bbox=matrix_transform(tmp_bbox,homoMatrix)
        tmp_bbox=new_tmp_bbox
#print("NEW TMP BBOX\n",new_tmp_bbox,new_tmp_bbox.shape,tmp_bbox,tmp_bbox.shape)
    if idx in range(idx_range):
        new_tmp_bbox_x1=tmp_bbox[0][0]
        new_tmp_bbox_y1=tmp_bbox[0][1]
        new_tmp_bbox_x2=tmp_bbox[1][0]
        new_tmp_bbox_y2=tmp_bbox[1][1]
        if new_nonZeroFList[idx][0] < new_tmp_bbox_x1 or new_nonZeroFList[idx][1]<new_tmp_bbox_y1 or new_nonZeroFList[idx][0]>new_tmp_bbox_x2 or new_nonZeroFList[idx][1]>new_tmp_bbox_y2:
            new_nonZeroFList[idx]=[0,0]

    new_bbox=new_tmp_bbox.reshape(1,2,2)
    features_fillzeros=np.zeros((FListNum,2))
    
    """
    remainNumOfFList, remainFList=extractNonZeroFeature(new_nonZeroFList)
    if remainNumOfFList < FListNum * 0.7:
        new_bbox = new_bbox.astype(int)
        new_features = getFeatures(frame, new_bbox)
        newFListNum,new_FList=extractFeaturefromFeatures(new_features)
        features_fillzeros[:newFListNum,:]=new_FList.copy()
        #print("NEWWBBOX/ FEATURE FILL ZEROS\n",features_fillzeros)
        print("NEW BBOX IS COMING/n/n/n/n")
    else:
    """
    features_fillzeros[:new_nonZeroFListNum,:]=new_nonZeroFList

    features_fillzeros=features_fillzeros.reshape(FListNum,1,-1)

#print("FEATURES FILLED UP WITH ZEROS",features_fillzeros)
    return features_fillzeros, new_bbox


