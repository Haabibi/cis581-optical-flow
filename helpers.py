import numpy as np
import cv2
import scipy

def interp2(v, xq, yq):
    """
    Description: Interpolation on 2D image
    Input:
        v: original value matrix, (H,W)
        xq: x coordinates of target points, (H,W)
        yq: y coordinates of target points, (H,W)
    Output:
        interp_val: target value matrix, (H,W)
    """
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()
    
    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'
        
    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def getWinBound(img_sz, startX, startY, win_size):
    """
    Description: Generate a window(patch) around the start point
    Input:
        img: Input image 2D shape, (2,)
        startX: start point x coordinate, Scalar
        startY: start point y coordinate, Scalar
        win_size: window size, Scalar
    Output:
        win_left: left bound of window, Scalar
        win_right: right bound of window, Scalar
        win_top: top bound of window, Scalar
        win_bottom: bottom bound of window, Scalar
    """
    szY, szX = img_sz
    
    win_left = startX - (win_size - 1) // 2
    win_right = startX + (win_size + 1) // 2
    if win_left < 0: win_left, win_right = 0, win_size
    elif win_right > szX: win_left, win_right = szX - win_size, szX
        
    win_top = startY - (win_size - 1) // 2
    win_bottom = startY + (win_size + 1) // 2
    if win_top < 0: win_top, win_bottom = 0, win_size
    elif win_bottom > szY: win_top, win_bottom = szY - win_size, szY

    return win_left, win_right, win_top, win_bottom

def findGradient(img):
    fx = np.array([[1,0, -1]])
    fy = fx.T
    Ix = scipy.signal.convolve2d(img, fx, 'same', 'symm')
    Iy = scipy.signal.convolve2d(img, fy, 'same', 'symm')
    return Ix, Iy

def optical_flow(img1, img2, Jx,Jy):
    It = img2 - img1
    A = np.hstack((Jx.reshape(-1, 1), Jy.reshape(-1, 1)))
    b = -It.reshape(-1, 1)
    res = np.linalg.solve(A.T @ A, A.T @ b)
    return res[0, 0], res[1, 0]

def calcJxJy(Ix,Iy,xx,yy):
    Jx=interp2(Ix,xx,yy)
    Jy=interp2(Iy,xx,yy)
    return Jx,Jy

def get_new_img(img, dx, dy):
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    new_x, new_y = x + dx, y + dy
    return interp2(img, new_x, new_y)


def extractFeaturefromFeatures(features):
    featureNum=features.shape[0]
    feature=features.reshape((featureNum,-1))
    return featureNum,feature

def extractNonZeroFeature(featureList):
    non_zero_mask=np.logical_or(featureList[:,0]>0,featureList[:,1]>0)
    nonZero_feature=featureList[non_zero_mask,:]
    nonZeroFeatureNum=nonZero_feature.shape[0]
    return nonZeroFeatureNum,nonZero_feature
