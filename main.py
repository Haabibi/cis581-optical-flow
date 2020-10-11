import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os

from optical_flow import *
"""
def objectTracking(rawVideo):
    cap = cv2.VideoCapture(rawVideo)
    imgs = []
    frame_cnt = 0 
                  
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        frame_to_show = frame.copy()
        print("frame:", frame_cnt)
        
        if frame_cnt == 1:
            x,y,w,h=np.int32(cv2.selectROI("roi",frame,fromCenter=False))
            print(x,y,w,h)
            bbox = np.zeros((1,2,2),dtype=int)
            bbox[:,0,0]=x
            bbox[:,0,1]=y
            bbox[:,1,0]=x+w
            bbox[:,1,1]=y+h
            initFeatureNum,features = getFeatures(frame, bbox)
            frame_old = frame.copy()
        
        else:
            if frame_cnt % 5 ==0:
                new_features = estimateAllTranslation(features, frame_old, frame)
                new_frame_to_show = frame.copy()
                new_features, bbox = applyGeometricTransformation( features, new_features, bbox)
                new_FListNum,new_FList=extractFeaturefromFeatures(new_features)
                remainNumOfFList, remainFList=extractNonZeroFeature(new_FList)
                
                if remainNumOfFList < initFeatureNum * 0.5:
                    tmp_bbox = bbox.astype(int)
                    x,new_features = getFeatures(frame, tmp_bbox)
                    newFListNum,new_FList=extractFeaturefromFeatures(new_features)
                    features_fillzeros=np.zeros((initFeatureNum,2))
                    features_fillzeros[:newFListNum,:]=new_FList.copy()
                    new_features=features_fillzeros.reshape(initFeatureNum,1,-1)
                    #print("NEWWBBOX/ FEATURE FILL ZEROS\n",features_fillzeros)
                    print("NEW BBOX IS COMING/n/n/n/n")
#                else:
#                    features_fillzeros=np.zeros((initFeatureNum,2))
#                    features_fillzeros[:remainNumOfFList,:]=remainFList.copy()
#                    new_features=features_fillzeros.reshape(initFeatureNum,1,-1)

                
                print("BBOX IN MAIN", bbox.shape)
                bbox = bbox.reshape(2,2)
                start_point = tuple(bbox[0].astype(int))
                end_point = tuple(bbox[1].astype(int))
                
                frame_old = frame.copy()
                vis = frame.copy()
                j=0
                tmp_new=np.reshape(new_features,(-1,2))
                for i in tmp_new:
                    x,y = i.ravel()
                    x = int(x)
                    y =int(y)
                    #print("X AND Y",x,y)
                    if x!=0 or y!=0:
                        cv2.circle(new_frame_to_show,(x,y),3,(255,255,255),2)
                        j+=1
                print("NUM OF FEATURES",j)
                cv2.rectangle(new_frame_to_show, start_point, end_point, (255,0,0), 3)
                cv2.imwrite("result_{}.jpg".format(frame_cnt), new_frame_to_show*255)
                
                features = new_features
                print("SHAPE",features.shape, new_features.shape)
                if(frame_cnt==100): #temp condition
                    break
                imgs.append(img_as_ubyte(vis))

            
            
        
        # save the video every 20 frames
        if frame_cnt % 20 == 0 or frame_cnt > 200 and frame_cnt % 10 == 0:
            imageio.mimsave('results/{}.gif'.format(frame_cnt), imgs)

"""
    

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
    trackVideo = 'results/Output_' + rawVideo
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
            print("Frame type",type(frame),type(frame_old),type(features))
            new_features = estimateAllTranslation(features, frame_old, frame)
            features, bbox = applyGeometricTransformation(features, new_features, bbox)
            frame_old = frame.copy()
            new_FListNum,new_FList=extractFeaturefromFeatures(new_features)
            remainNumOfFList, remainFList=extractNonZeroFeature(new_FList)
            
            if remainNumOfFList < initFeatureNum * 0.5:
                tmp_bbox = bbox.astype(int)
                x,new_features = getFeatures(frame, tmp_bbox)
                newFListNum,new_FList=extractFeaturefromFeatures(new_features)
                features_fillzeros=np.zeros((initFeatureNum,2))
                features_fillzeros[:newFListNum,:]=new_FList.copy()
                new_features=features_fillzeros.reshape(initFeatureNum,1,-1)
                #print("NEWWBBOX/ FEATURE FILL ZEROS\n",features_fillzeros)
                print("NEW BBOX IS COMING/n/n/n/n")
                features=new_features

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
