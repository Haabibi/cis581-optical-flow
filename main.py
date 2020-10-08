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
                  
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        frame_cnt += 1
        print("frame:", frame_cnt)
        
        if frame_cnt == 1:
            bbox = None
            features = getFeatures(frame, bbox)
            frame_old = frame.copy()

        else:
            new_features = estimateAllTranslation(features, frame_old, frame)
            features, bbox = applyGeometricTransformation(features, new_features, bbox)
            frame_old = frame.copy()
            vis = frame.copy()

            
            """ 
            TODO: Plot feature points and bounding boxes on vis
            """
            
            imgs.append(img_as_ubyte(vis))
        
        # save the video every 20 frames
        if frame_cnt % 20 == 0 or frame_cnt > 200 and frame_cnt % 10 == 0:
            imageio.mimsave('results/{}.gif'.format(frame_cnt), imgs)


if __name__ == "__main__":
    rawVideo = "test_videos/Easy.mp4"
    if not os.path.exists("results"): os.mkdir("results")
    objectTracking(rawVideo)
    



