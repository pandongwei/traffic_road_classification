#!/usr/bin/env python3
'''
This file is to test the result of the BOW methods
The 'TODO' part needs to be changed
PS: to extract SIFT/SURF feature, you need to install opencv < 3.4.1
You also need to uncommand the DETECTOR line in params.py file
'''

import cv2
from utils import *
import numpy as np
import params
from joblib import dump
#import graphviz
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier,BaseEnsemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
#from xgboost import XGBClassifier,XGBRFClassifier

################################################################################

def generate_dictionary(imgs_data, dictionary_size):   #dictionary_size=512

    # Extracting descriptors
    desc = stack_array([img_data.bow_descriptor for img_data in imgs_data])   #把imgs-data 中的bow_descriptor堆到desc中，组成一个数组

    # important, cv2.kmeans() clustering only accepts type32 descriptors

    desc = np.float32(desc)

    # perform clustering - increase iterations and reduce EPS to change performance

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, params.BOW_clustering_iterations, 0.01) #0.01
    flags = cv2.KMEANS_PP_CENTERS   #With this flag enabled, the method always starts with a random set of initial samples, and tries to converge

    # desc is a type32 numpy array of vstacked descriptors

    #compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    np.save(params.BOW_DICT_PATH, dictionary)

    return dictionary


def main():
    

    ############################################################################
    # load our training data set of images examples

    program_start_train = cv2.getTickCount()

    print("Loading images...")
    start = cv2.getTickCount()
    # TODO
    path, class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-train")

    # build a lisyt of class names automatically from our dictionary of class (name,number) pairs

    train_imgs_data = load_images(path, class_names)

    print(" training step with {} images".format(len(train_imgs_data)))
    print_duration(start)

    ############################################################################
    # perform bag of visual words feature construction

    print("Computing descriptors...") # for each training image
    start = cv2.getTickCount()   #开始计时
    [train_img_data.compute_bow_descriptor() for train_img_data in train_imgs_data]
    print_duration(start)

    
    print("Clustering...")          # over all images to generate dictionary code book/words
    start = cv2.getTickCount()
    dictionary = generate_dictionary(train_imgs_data, params.BOW_dictionary_size)
    #dictionary = np.load(multi_class_params.BOW_DICT_PATH)
    print_duration(start)

    print("Generating histograms...") # for each training image
    start = cv2.getTickCount()
    [train_img_data.generate_bow_hist(dictionary) for train_img_data in train_imgs_data]
    print_duration(start)

    ############################################################################
    # train an SVM based on these norm_features

    print("Training classifier...")
    start = cv2.getTickCount()

    #clf = BaggingClassifier(base_estimator=SVC(probability = True,gamma='scale',tol = 1e-5,C = 10),n_estimators=10,bootstrap_features=True,n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1)
    #clf = GradientBoostingClassifier(n_estimators=100)
    #estimator_list = [('bagging',clf1),('rf',clf2),('gbc',clf3)]
    #clf = VotingClassifier(estimators=estimator_list,voting='hard',n_jobs=-1)
    #clf = AdaBoostClassifier(base_estimator=SVC(probability = True,tol = 1e-5,gamma='scale',C = 1),n_estimators=20)
    #clf = AdaBoostClassifier(base_estimator=SVC(tol=1e-1, gamma='scale', C=1), n_estimators=50,algorithm='SAMME')
    clf = SVC(probability = True,tol = 1e-5,gamma='scale',C = 1)

    #clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=1000,activation='relu')
    #clf = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(200,),max_iter=1000,activation='relu'),n_estimators=10,bootstrap_features=True,n_jobs=-1)
    #param_grid = {'n_estimators':[10,50,100,500,1000],'bootstrap_features':['True','False'],}

    #clf = SGDClassifier()
    #clf = AdaBoostClassifier(base_estimator=SGDClassifier(),n_estimators=100,algorithm='SAMME')
    #clf = XGBClassifier(n_jobs=-1)
    #clf = XGBRFClassifier(max_depth=10,n_jobs=-1,n_estimators=1000)

    #clf = DecisionTreeClassifier()
    #clf = SVC(probability=True, gamma='scale', tol=1e-5,C = 1)
    #clf = BaggingClassifier(base_estimator=SVC(probability=True,gamma='scale', tol=1e-5,C = 10),n_estimators=10)

    samples = get_bow_histograms(train_imgs_data)    #提取出histogram
    # reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    d2_train_dataset = samples.reshape((nsamples,nx*ny))

    # get class label for each training image
    class_labels = get_class_labels(train_imgs_data)
    class_labels = class_labels.ravel()

    clf.fit(d2_train_dataset,class_labels)

    #predict_prob = clf.predict_proba(d2_train_dataset)

    result = clf.score(d2_train_dataset,class_labels)
    # TODO
    dump(clf, 'train_result/hog_rf.joblib')
    #error = errorCalculation(class_labels.ravel(),result)
    print(result)


    print_duration(start)
    print(("Finished training part. It took {}".format(format_time(get_elapsed_time(program_start_train)))))

    #input("Press Enter to continue...")

    # testing part
    print("Start testing ....")

    program_start_test = cv2.getTickCount()
    start = cv2.getTickCount()
    # TODO
    path, class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-test")

    test_imgs_data = load_images(path, class_names)
    print("testing with {} test-images".format(len(test_imgs_data)))

    print_duration(start)

    print("Computing descriptors...") # for each training image
    start = cv2.getTickCount()   #开始计时
    [test_img_data.compute_bow_descriptor() for test_img_data in test_imgs_data]
    print_duration(start)
 
    print("Generating histograms...") # for each testing image
    start = cv2.getTickCount()
    [test_img_data.generate_bow_hist(dictionary) for test_img_data in test_imgs_data]
    print_duration(start)   
    samples = get_bow_histograms(test_imgs_data)
    #数据降维 reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    d2_test_dataset = samples.reshape((nsamples,nx*ny))
    class_labels = get_class_labels(test_imgs_data)
    class_labels = class_labels.ravel()

    print("Performing batch SVM classification over all data  ...")
    start = cv2.getTickCount()#
    result = clf.score(d2_test_dataset,class_labels)
    print_duration(start)
    print(result)
    print_duration(program_start_test)

    output = clf.predict(d2_test_dataset)
    error, missclass_list = errorCalculation_return_missclassList(class_labels, output)
    print("With affilation, the classifier got {}% of the testing examples correct!".format(
        round((1.0 - error) * 100, 2)))

################################################################################

if __name__ == '__main__':
    main()

################################################################################
