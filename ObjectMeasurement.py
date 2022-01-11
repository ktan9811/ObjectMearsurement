
import cv2
import numpy as np
import preproc
import usermath

fx = 1493.187892
fy = 1493.187892
cx = 720.000000
cy = 540.000000
k1 = 0.084614
k2 = -0.371386
p1 = -0.015346
p2 = 0.018616

mtx=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
dist=np.array([k1,k2,p1,p2,0])


pixel_len=None
##mouse를 통한 최근접 윤곽선 검출 및 기준 물체 가로 길이 입력
def mouse_call_back(event, x, y, flags, param):
    global pixel_len                                                #전역변수 pixel_len 가져오기
    if event == cv2.cv2.EVENT_LBUTTONDOWN:
        for i in range(0, len(contours)):                           #검출된 모든 contour들에 대해 x,y좌표가 내부에 있는지 확인
            r = cv2.pointPolygonTest(contours[i], (x, y), False)
            if r > 0:                                               #내부에 있을시 Selected 됨을 알려줌
                print("Selected contour ", i)
                known_width=input("클릭한 물체의 너비를 입력하세요 ex)단위:cm\n")                #클릭한 contour의 물체의 너비 입력받고 known_width에 넣기
                pixel_len=W_init[i]/float(known_width)              #pixel 길이 계산 

##메인 함수
src = cv2.imread('not90_2.jpg')
src = cv2.resize(src, (0, 0), None, 0.75, 0.75)
# camera=np.load('calib.npz')     #calibration 계수들 불러오기
# mtx,dist=camera['mtx'],camera['dist']

h,  w = src.shape[:2]
newmtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


dst = cv2.undistort(src, mtx, dist, None, newmtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
dst, data = preproc.getContours(src, (25,100), 1000)                #함수 return 값을 dst, data에 저장
contours = []
W_init=[]         #pixel_len 계산할 때 index에 해당하는 contour의 width 꺼내와야 하므로 list로 선언 
for i in range(0, len(data)):                                       #contours에 data[][4]에 저장된 contours  복사
    contours.append(data[i][4])

#마우스 클릭시 어떤 contour 내부의 점이면 contour number(크기순) 반환, c입력시 break
while True:
    cv2.setMouseCallback('objmeasurement', mouse_call_back)
    dst1=dst.copy()             #기준 물체 선정할 때마다 길이 새로 측정해야 하므로 copy
    if len(data) != 0:
        for i,obj in enumerate(data):
            orderpts=usermath.reorder(obj[2])           #꼭짓점 점들 정렬
            W_init.append(usermath.euclidean(orderpts[0][0],orderpts[1][0]))  #width를 euclidean distance로 계산
            H_init=usermath.euclidean(orderpts[0][0],orderpts[2][0])          #height를 euclidean distance로 계산
            if pixel_len is not None:               #pixel 길이가 측정되었다면
                W=round((W_init[i]/pixel_len),2)        # Width 예측하고 반올림
                H=round((H_init/pixel_len),2)           # height 예측하고 반올림
                x,y,w,h=obj[3]
                dst1=cv2.putText(dst1,'{}'.format(W),(x+w//2-30,y+10),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2) #가로 길이 이미지에 표시
                dst1=cv2.putText(dst1,'{}'.format(H),(x,y+h//2),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2) #세로 길이 이미지에 표시
    cv2.imshow('objmeasurement',dst1)
    k = cv2.waitKey(1)

    if k == ord('c'):
        break
cv2.destroyAllWindows()