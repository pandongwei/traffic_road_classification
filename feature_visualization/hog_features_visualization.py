'''
提取一张图片的hog feature并可视化在图片上
'''

from skimage.feature import hog
import cv2

im = cv2.imread('00454.png')
#im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
normalised_blocks, hog_image = hog(im, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True,multichannel=True)
hog_image = hog_image*5  # this command is to adjust the significance of the hog feature
cv2.imshow('HOG',hog_image)
cv2.imwrite('result/HOG1.jpg',hog_image)
cv2.waitKey(0)
if cv2.waitKey(1) and 0xFF == ord('q'):
    cv2.destroyAllWindows()