"""Suppress camera-moving scenery using adjacent-frame affine alignment."""
import cv2
import numpy as np


def newly_brightened(frames):
    result = [np.zeros_like(frames[0])]
    for previous, frame in zip(frames, frames[1:]):
        before = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(before, maxCorners=160, qualityLevel=.01, minDistance=5)
        warped = previous
        if points is not None and len(points) >= 8:
            following, status, _ = cv2.calcOpticalFlowPyrLK(
                before, after, points, None, winSize=(15, 15), maxLevel=2)
            if following is not None and status.sum() >= 8:
                valid = status.ravel().astype(bool)
                cv2.setRNGSeed(0)
                transform, _ = cv2.estimateAffinePartial2D(
                    points[valid], following[valid], method=cv2.RANSAC, ransacReprojThreshold=2)
                if transform is not None:
                    warped = cv2.warpAffine(previous, transform, (frame.shape[1], frame.shape[0]),
                                           borderMode=cv2.BORDER_REFLECT)
        difference = frame.astype(float) - warped
        keep = (difference.sum(axis=2) > 40) & (difference.max(axis=2) > 20)
        result.append(np.where(keep[:, :, None], frame, 0).astype('uint8'))
    return result
