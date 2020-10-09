import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os

from optical_flow_abby import *

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
                  
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        print("frame:", frame_cnt)
        
        if frame_cnt == 1:
            x,y,w,h=np.int32(cv2.selectROI("roi",frame,fromCenter=False))
            print(x,y,w,h)
            bbox = np.zeros((1,2,2),dtype=int)
            bbox[:,0,0]=x
            bbox[:,0,1]=y
            bbox[:,1,0]=x+w
            bbox[:,1,1]=y+h
            print(bbox)
            features = getFeatures(frame, bbox)
            frame_old = frame.copy()
        
        else:
            if frame_cnt % 5 ==0:
                new_features = estimateAllTranslation(features, frame_old, frame)
                for i in new_features:
                    x,y = i.ravel()
                    x = int(x)
                    y =int(y)
                    cv2.circle(frame,(x,y),3,(0,0,255),3)
                
                cv2.imwrite("result{}.jpg".format(frame_cnt),frame*255)
                features, bbox = applyGeometricTransformation(features, new_features, bbox)
                frame_old = frame.copy()
                vis = frame.copy()
                #cv2.imwrite
                features = new_features
                if(frame_cnt==20): #temp condition
                    break
                imgs.append(img_as_ubyte(vis))
            """ 
            TODO: Plot feature points and bounding boxes on vis
            """
            
            
        
        # save the video every 20 frames
        if frame_cnt % 20 == 0 or frame_cnt > 200 and frame_cnt % 10 == 0:
            imageio.mimsave('results/{}.gif'.format(frame_cnt), imgs)


if __name__ == "__main__":
    rawVideo = "test_videos/Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(rawVideo)
    

