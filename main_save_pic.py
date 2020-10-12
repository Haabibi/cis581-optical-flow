import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os

from optical_flow import *
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
        H,W = frame_to_show.shape
        if frame_cnt == 120:
            x,y,w,h=np.int32(cv2.selectROI("roi",frame,fromCenter=False))
            print(x,y,w,h)
            bbox = np.zeros((1,2,2),dtype=int)
            bbox[:,0,0]=x
            bbox[:,0,1]=y
            bbox[:,1,0]=x+w
            bbox[:,1,1]=y+h
            initFeatureNum,features = getFeatures(frame, bbox)
            frame_old = frame.copy()
        
        elif frame_cnt>120:
            if frame_cnt % 1 ==0:
                new_features = estimateAllTranslation(features, frame_old, frame)
                new_frame_to_show = frame.copy()
                
                new_features, tmp_bbox = applyGeometricTransformation(features, new_features, bbox,H,W)
                new_FListNum,new_FList=extractFeaturefromFeatures(new_features)
                remainNumOfFList, remainFList=extractNonZeroFeature(new_FList)

                #print("TMPBBOX\n",tmp_bbox)
                bbox=tmp_bbox
                #print("BBOX IN MAIN", bbox)
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
                        cv2.circle(new_frame_to_show,(x,y),3,(255,255,255),-1)
                        j+=1
                #print("NUM OF FEATURES",j)
                cv2.rectangle(new_frame_to_show, start_point, end_point, (255,0,0), 2)
                cv2.imwrite("result_{}.jpg".format(frame_cnt), new_frame_to_show*255)
                
                if remainNumOfFList < initFeatureNum * 0.6:
                    break
#                    int_bbox = bbox.astype(int)
#                    int_bbox=int_bbox.reshape((1,2,2))
#                    #print("NEWWBBOX\n",int_bbox)
#                    x,new_features = getFeatures(frame, int_bbox)
#                    newFListNum,new_FList=extractFeaturefromFeatures(new_features)
#                    features_fillzeros=np.zeros((initFeatureNum,2))
#                    features_fillzeros[:newFListNum,:]=new_FList.copy()
#                    new_features=features_fillzeros.reshape(initFeatureNum,1,-1)

                    #print("NEW BBOX IS COMING/n/n/n/n")
                
                else:
                    bbox=tmp_bbox
                
                features = new_features
                print("SHAPE",features.shape, new_features.shape)
                if(frame_cnt==100): #temp condition
                    break
                imgs.append(img_as_ubyte(vis))
        # save the video every 20 frames
        #if frame_cnt % 20 == 0 or frame_cnt > 200 and frame_cnt % 10 == 0:
            # imageio.mimsave('results/{}.gif'.format(frame_cnt), imgs)



if __name__ == "__main__":
    rawVideo = "./test_videos/Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(rawVideo)
