#!/usr/bin/env python3

import cv2
import os

################################################################################
# settings for datsets in general

master_path_to_dataset = "dataset-sample" # ** need to edit this **就是选择data所在的文件夹

# data location - training examples

DATA_training_path_bicycle_lane = os.path.join(master_path_to_dataset,"pedestrian/")
DATA_training_path_bicycle_lane_and_pedestrian = os.path.join(master_path_to_dataset,"car-lane/")
DATA_training_path_car_and_bicycle_lane = os.path.join(master_path_to_dataset,"bicycle-lane-and-pedestrian/")
DATA_training_path_car_lane = os.path.join(master_path_to_dataset,"car-and-bicycle-lane/")
DATA_training_path_pedestrian = os.path.join(master_path_to_dataset,"bicycle-lane/")

DATA_WINDOW_SIZE = [64, 128]
#DATA_WINDOW_SIZE = [512, 256] #used in hog and 32parts

DATA_CLASS_NAMES = {
    "bicycle-lane": 0,
    "bicycle-lane-and-pedestrian": 1,
    "car-lane": 2,
    "pedestrian": 3
}
DATA_CLASS_NAMES_in_vector = {
    "bicycle-lane": [1,0,0,0],
    "bicycle-lane-and-pedestrian": [0,1,0,0],
    "car-lane": [0,0,1,0],
    "pedestrian": [0,0,0,1]
}
################################################################################
# settings for BOW - Bag of (visual) Word - approaches

BOW_SVM_PATH = "train_result/svm_bow.xml"
BOW_DICT_PATH = "train_result/bow_dictionary.npy"

BOW_dictionary_size = 512  #  in general, larger = better performance, but potentially slower，512
BOW_SVM_kernel = cv2.ml.SVM_RBF  # see opencv manual for other options
BOW_SVM_max_training_iterations = 1000 # stop training after max iterations 500

BOW_clustering_iterations = 20 # reduce to improve speed, reduce quality 20

BOW_fixed_feature_per_image_to_use = 200 # reduce to improve speed, set to 0 for variable number


# TODO: the opencv higher than 3.4.1 doesn't have SIFT feature any more cause it is not free.
# TODO: so in order to use BOW method, you need to install opencv < 3.4.1
# DETECTOR = cv2.xfeatures2d.SIFT_create(BOW_fixed_feature_per_image_to_use)  # -- requires extra modules and non-free build flag
# DETECTOR = cv2.xfeatures2d.SURF_create(hessianThreshold = 100,nOctaveLayers=5,nOctaves=5,extended=True) # -- requires extra modules and non-free build flag


# as SIFT/SURF feature descriptors are floating point use KD_TREE approach

_algorithm = 0 # FLANN_INDEX_KDTREE
_index_params = dict(algorithm=_algorithm, trees=5)
_search_params = dict(checks=50)           #这几行代码跟opencv网上例子一样




MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)   #用于两图匹配对应点

################################################################################
# settings for HOG approaches

HOG_SVM_PATH = "train_result/svm_hog.xml"

HOG_SVM_kernel = cv2.ml.SVM_LINEAR # see opencv manual for other options
HOG_SVM_max_training_iterations = 1000 # stop training after max iterations

################################################################################
