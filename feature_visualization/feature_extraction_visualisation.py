'''
feature visualization of different features
'''
import cv2
import numpy as np

image = cv2.imread('00454.png')
#SIFT
sift_features = cv2.xfeatures2d.SIFT_create().detect(image)
img_sift = cv2.drawKeypoints(image,sift_features,np.array([]),(0,0,255),flags =0)
#SURF
surf_features = cv2.xfeatures2d.SURF_create().detect(image)
img_surf = cv2.drawKeypoints(image,surf_features,np.array([]),(0,0,255),flags =0)
#MSER
mser_featrues = cv2.MSER_create().detect(image)
img_mser = cv2.drawKeypoints(image,mser_featrues,np.array([]),(0,0,255),flags =0)
#FastFeatureDetect
ffd_features = cv2.FastFeatureDetector_create().detect(image)
img_ffd = cv2.drawKeypoints(image,ffd_features,np.array([]),(0,0,255),flags =0)
#ORB
orb_features = cv2.ORB_create().detect(image)
img_orb = cv2.drawKeypoints(image,orb_features,np.array([]),(0,0,255),flags =0)
#HarrisLaplaceFeatureDetector
hlfd_features = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create().detect(image)
img_hlfd = cv2.drawKeypoints(image,hlfd_features,np.array([]),(0,0,255),flags =0)
#VGG
#vgg_features = cv2.xfeatures2d.VGG_create().detect(image)
#img_vgg = cv2.drawKeypoints(image,vgg_features,np.array([]),(0,0,255),flags =0)



cv2.imwrite('result/sift1.jpg',img_sift)
cv2.imwrite('result/surf1.jpg',img_surf)
cv2.imwrite('result/mser1.jpg',img_mser)
cv2.imwrite('result/FastFeatureDetect1.jpg',img_ffd)
cv2.imwrite('result/orb1.jpg',img_orb)
cv2.imwrite('result/HarrisLaplaceFeatureDetector1.jpg',img_hlfd)

cv2.imshow('origin',image)
cv2.imshow('sift',img_sift)
cv2.imshow('surf',img_surf)
cv2.imshow('mser',img_mser)
cv2.imshow('FastFeatureDetect',img_ffd)
cv2.imshow('orb',img_orb)
cv2.imshow('HarrisLaplaceFeatureDetector',img_hlfd)


cv2.waitKey(0)
if cv2.waitKey(1) and 0xFF == ord('q'):
    cv2.destroyAllWindows()                        
