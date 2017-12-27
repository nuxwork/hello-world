import os
import sys
import cv2
import dlib

def pwd():
	if getattr(sys, 'frozen', False):
		path = os.path.dirname(sys.executable)
	elif __file__:
		path = os.path.dirname(__file__)
	return os.path.abspath(path)

def mmm():
# dlib人脸关键点检测器

    predictor_path = "shape_predictor_5_face_landmarks.dat"

    predictor = dlib.shape_predictor(predictor_path)  

 

    # dlib正脸检测器

    detector = dlib.get_frontal_face_detector()

 

    # 正脸检测

    dets = detector(img, 1)

 

    # 如果检测到人脸

    if len(dets)>0:  

        for d in dets:

            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()

            # x,y,w,h = faceRect  

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)

 

            # 关键点检测，5个关键点

            shape = predictor(img, d)

            for point in shape.parts():

                cv2.circle(img,(point.x,point.y),3,color=(0,255,0))

 

            cv2.imshow("image",img)

            cv2.waitKey()  

def main():
    hat_img = os.path.abspath( pwd() + '/yard.jpeg' )
    hat_img = cv2.imread(hat_img)
    (b,g,r) = cv2.split(hat_img)
    cv2.imshow("red", r)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

    # mmm()