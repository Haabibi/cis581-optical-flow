import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os

from optical_flow import *
def objectTracking(rawVideo):
    """

        Description: Generate and save tracking video
        Input:
        rawVideo: Raw video file name, String
        Instruction: Please feel free to use cv.selectROI() to manually select bounding box

    """
    cap = cv2.VideoCapture(rawVideo)
    imgs = []
    frame_cnt = 0
    
    # Initialize video writer for tracking video
    trackVideo = './results/Output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    
    # Define how many objects to track
    F = 1
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: continue
        vis = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        H,W = frame.shape
        if frame_cnt == 1:
            #bbox = np.zeros((F,2,2))
            bbox = np.zeros((F,2,2),dtype=int)
            # Manually select objects on the first frame
            for f in range(F):
                x,y,w,h = np.int32(cv2.selectROI("roi", vis, fromCenter=False))
                cv2.destroyAllWindows()
                
                bbox[:,0,0]=x
                bbox[:,0,1]=y
                bbox[:,1,0]=x+w
                bbox[:,1,1]=y+h
            
            initFeatureNum,features = getFeatures(frame, bbox)
            frame_old = frame.copy()
            new_FListNum,new_FList=extractFeaturefromFeatures(features)
            
        else:
            #print("Frame type",type(frame),type(frame_old),type(features))
            new_features = estimateAllTranslation(features, frame_old, frame)
            features, tmp_bbox = applyGeometricTransformation(features, new_features, bbox,H,W)
            frame_old = frame.copy()
            new_FListNum,new_FList=extractFeaturefromFeatures(new_features)
            remainNumOfFList, remainFList=extractNonZeroFeature(new_FList)
            
            
            if remainNumOfFList < initFeatureNum * 0.6:
                bbox_w=bbox[:,1,0]-bbox[:,0,0]
                bbox_h=bbox[:,1,1]-bbox[:,0,1]
                print("BBOX\n",bbox, bbox_w,bbox_h)
                if bbox_w<30 or bbox_h <30:
                    print("bbox too small")
                    break
                elif bbox[:,1,0]==W or bbox[:,1,1]==H:
                    print("bbox out of bound")
                    break
                int_bbox = bbox.astype(int)
                int_bbox=int_bbox.reshape((F,2,2))
                
                x,new_features = getFeatures(frame, int_bbox)
                newFListNum,new_FList=extractFeaturefromFeatures(new_features)
                features_fillzeros=np.zeros((initFeatureNum,2))
                features_fillzeros[:newFListNum,:]=new_FList.copy()
                new_features=features_fillzeros.reshape(initFeatureNum,1,-1)
            else:
                bbox=tmp_bbox

        # # display the bbox
        for f in range(F):
             cv2.rectangle(vis, tuple(bbox[f,0].astype(np.int32)), tuple(bbox[f,1].astype(np.int32)), (0,0,255), thickness=2)
        
        # # display feature points
        for f in range(F):
             for feat in new_FList:
                 cv2.circle(vis, tuple(feat.astype(np.int32)), 2, (0,255,0), thickness=-1)


        # save to list
        imgs.append(img_as_ubyte(vis))
        
        # save image
        if (frame_cnt + 1) % 5 == 0:
            cv2.imwrite('results/{}.jpg'.format(frame_cnt), img_as_ubyte(vis))
        
        # Save video with bbox and all feature points
        writer.write(vis)
        
        # Press 'q' on the keyboard to exit
        cv2.imshow('Track Video', vis)
        if cv2.waitKey(30) & 0xff == ord('q'): break
    
    
    # Release video reader and video writer
    cv2.destroyAllWindows()
    cap.release()
    writer.release()
    
    return


if __name__ == "__main__":
    rawVideo = "./test_videos/Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(rawVideo)
