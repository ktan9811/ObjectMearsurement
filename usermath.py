import cv2
import numpy as np


def reorder(points):                #contour 점들을 정렬시키는 함수
    dst_points=np.zeros_like(points)
    points=points.reshape((-1,2))
    sum=points.sum(1)
    dst_points[0]=points[np.argmin(sum)]    #(오른쪽 위는 0번째 element)
    dst_points[3]=points[np.argmax(sum)]    #(오른쪽 아래는 3번째 element)
    diff=np.diff(points,axis=1)
    dst_points[1]=points[np.argmin(diff)]   #(왼쪽 위는 1번째 element)
    dst_points[2]=points[np.argmax(diff)]   #(왼쪽 아래는 2번째 element)
    return dst_points

def euclidean(pts1,pts2):
    return ((pts2[0]-pts1[0])**2+(pts2[1]-pts1[1])**2)**0.5

    