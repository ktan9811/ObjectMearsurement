import cv2
import numpy as np

##윤곽선 검출 input : img(bgr), T[Low:High], 최소 영역크기  ## output : contour + img, data[len, area, apporx, box, contour]
def getContours(img, T = [50,100], minarea = 100):
    dst = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1)
    edged = cv2.Canny(gray, T[0], T[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    edged = cv2.dilate(edged, kernel, iterations=3)
    edged = cv2.erode(edged, kernel, iterations=2)

    #윤곽선 검출
    contours, hier  = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    data = []       #return값을 저장할 data변수 생성

    for i  in contours:
        area = cv2.contourArea(i)
        if area > minarea:                                          #최소 영역보다 큰 area들에 한하여 시행 (노이즈제거)
            length = cv2.arcLength(i, True)                         #길이구해서
            approx = cv2.approxPolyDP(i, 0.02*length, True)         #유사 도형 검출
            box    = cv2.boundingRect(approx)
            data.append([len(approx), area, approx, box, i])        #데이터 저장

    for i in data:
        cv2.drawContours(dst, i[4], -1, (255,0,0), 2)               #dst이미지에 검출된 contour 그림
    return dst, data