'''
this file tries to concat the hog and sift feature together to improve the performance(failed)
PS: to extract SIFT/SURF feature, you need to install opencv < 3.4.1
You also need to uncommand the DETECTOR line in params.py file
'''

from utils import *
from joblib import dump
import cv2
from kalman_filter import *
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
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

from sklearn import tree

def generate_dictionary(imgs_data, dictionary_size):   #dictionary_size=512

    # Extracting descriptors
    desc = stack_array([img_data.bow_descriptor for img_data in imgs_data])   #把imgs-data 中的bow_descriptor堆到desc中，组成一个数组

    # important, cv2.kmeans() clustering only accepts type32 descriptors

    desc = np.float32(desc)

    # perform clustering - increase iterations and reduce EPS to change performance

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, params.BOW_clustering_iterations, 0.01) #0.01
    flags = cv2.KMEANS_PP_CENTERS   #With this flag enabled, the method always starts with a random set of initial samples, and tries to converge


    compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    np.save(params.BOW_DICT_PATH, dictionary)
    return dictionary

def concat_hog_sift(imgs_data):
    samples = stack_array([[np.vstack((img_data.hog_descriptor,img_data.bow_histogram))] for img_data in imgs_data])
    return np.float32(samples)

def main():
    
    ############################################################################
    # load our training data set of images examples

    program_start = cv2.getTickCount()

    print("Loading images...")
    start = cv2.getTickCount()

    paths,class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-train")
    train_imgs_data = load_images(paths, class_names)
    '''
    paths_extra,class_names_extra = generate_path("/home/pan/master-thesis-in-mrt/data/dataset-evaluation")
    train_imgs_data_extra = load_images(paths_extra, class_names_extra)
    train_imgs_data.extend(train_imgs_data_extra)
    train_imgs_data = shuffle(train_imgs_data)
    '''
    print(("Loaded totally {} image(s)".format(len(train_imgs_data))))
    print_duration(start)

    ############################################################################
    print("Computing SIFT descriptors...")  # for each training image
    start = cv2.getTickCount()  # 开始计时
    [train_img_data.compute_bow_descriptor() for train_img_data in train_imgs_data]
    print_duration(start)

    print("Clustering...")  # over all images to generate dictionary code book/words
    start = cv2.getTickCount()
    dictionary = generate_dictionary(train_imgs_data, params.BOW_dictionary_size)
    # dictionary = np.load(multi_class_params.BOW_DICT_PATH)
    print_duration(start)

    print("Generating histograms...")  # for each training image
    start = cv2.getTickCount()
    [train_img_data.generate_bow_hist(dictionary) for train_img_data in train_imgs_data]
    print_duration(start)

    print("Computing HOG descriptors...") # for each training image
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in train_imgs_data]
    print_duration(start)

    ############################################################################
    # train an SVM based on these norm_features

    print("Training Classifier...")
    start = cv2.getTickCount()

    # compile samples (i.e. visual word histograms) for each training image

    samples = concat_hog_sift(train_imgs_data)    #提取出histogram
    #数据降维 reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    d2_train_dataset = samples.reshape((nsamples,nx*ny))

    # get class label for each training image
    class_labels = get_class_labels(train_imgs_data)
    class_labels = class_labels.ravel()

    # select the classifiers
    #clf = BaggingClassifier(base_estimator=SVC(probability = True,gamma='scale',tol = 1e-5,C = 10), n_estimators=10, bootstrap=True, bootstrap_features=True, n_jobs=10)
    #clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1)
    #clf = GradientBoostingClassifier(n_estimators=100)
    #estimator_list = [('bagging',clf1),('rf',clf2),('gbc',clf3)]
    #clf = VotingClassifier(estimators=estimator_list,voting='hard',n_jobs=-1)
    #clf = AdaBoostClassifier(base_estimator=SVC(probability = True,tol = 1e-5,gamma='scale',C = 1),n_estimators=20)
    clf = SVC(probability = True,tol = 1e-5,gamma='scale',C = 10)

    #clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=1000,activation='relu')
    #clf = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(200,),max_iter=1000,activation='relu'),n_estimators=10,bootstrap_features=True,n_jobs=1)
    #param_grid = {'n_estimators':[10,50,100,500,1000],'bootstrap_features':['True','False'],}

    #clf = SGDClassifier()
    #clf = AdaBoostClassifier(base_estimator=SGDClassifier(),n_estimators=100,algorithm='SAMME')
    #clf = XGBClassifier(n_jobs=-1)
    #clf = XGBRFClassifier(max_depth=10,n_jobs=-1,n_estimators=1000)

    #clf = GridSearchCV(BaggingClassifier(base_estimator=SVC(probability=True, gamma='scale', tol=1e-5,C =10),n_jobs=60  ),param_grid,cv = 5)


    clf.fit(d2_train_dataset,class_labels)

    #predict_prob = clf.predict_proba(d2_train_dataset)
    result = clf.score(d2_train_dataset,class_labels)
    dump(clf, 'train_result/train_whole_dataset_bagging.joblib')

    #error = errorCalculation(class_labels.ravel(),result)
    print(result)
    #print("SVM got {}% of the training examples correct!".format(round((1.0 - error) * 100,2)))
    #visualize the tree

    print_duration(start)

    print(("Finished training HOG detector. {}".format(format_time(get_elapsed_time(program_start)))))

    #input("Press Enter to continue...")

    #测试部分开始
    print("Start testing ....")

    start = cv2.getTickCount()

    path, class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-test")

    test_imgs_data = load_images(path, class_names)
    #test_imgs_data = shuffle(test_imgs_data)
    print("testing with {} test-images".format(len(test_imgs_data)))
    print_duration(start)

    print("Computing SIFT descriptors...")  # for each training image
    start = cv2.getTickCount()  # 开始计时
    [test_img_data.compute_bow_descriptor() for test_img_data in test_imgs_data]
    print_duration(start)

    print("Generating histograms...")  # for each training image
    start = cv2.getTickCount()
    [test_img_data.generate_bow_hist(dictionary) for test_img_data in test_imgs_data]
    print_duration(start)

    [img_data.compute_hog_descriptor() for img_data in test_imgs_data]

    samples = concat_hog_sift(test_imgs_data)
    #数据降维 reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    d2_test_dataset = samples.reshape((nsamples,nx*ny))
    class_labels = get_class_labels(test_imgs_data)
    class_labels = class_labels.ravel()

    print("Performing batch  classification over all data  ...")
    start = cv2.getTickCount()
    result = clf.score(d2_test_dataset,class_labels)

    print_duration(start)
    print('the test result is: ',result)

    #combine kalman filter
    predict_prob = clf.predict_proba(d2_test_dataset)
    predict_prob = predict_prob.T
    output_kalman = []
    x_state = KalmanFilter()
    for i in range(len(predict_prob[0])):
        #kalman part
        x_state.predict()
        x_state.correct(predict_prob[:,i])

        x_combination = x_state.x_e
        output_kalman.append(np.argmax(x_combination))

    # f = open("result_summery.txt", 'a+')
    error_kalman, missclass_list_kalman = errorCalculation_return_missclassList(class_labels, output_kalman)
    print("after combining kalman filter the classifier got {}% of the testing examples correct!".format(round((1.0 - error_kalman) * 100, 2)))
    #write the result in the txt
    #print('the classifier is: ', clf, file=f)
    #print('test accuracy without kalman 4 classes: {}%'.format(round(result*100, 2)), file=f)
    #print('test accuracy with kalman 4 classes: {}%'.format(round((1.0 - error_kalman) * 100, 2)), file=f)
    #print(missclass_list_kalman,file = f)

    #add affilation
    error_affilation, missclass_list_affilation, _ = errorCalculation_affilation_return_missclassList(class_labels, output_kalman)
    print("after affilation to 3 classes the classifier got {}% of the testing examples correct!".format(round((1.0 - error_affilation) * 100, 2)))
    #print('test accuracy with kalman 3 classes: {}%'.format(round((1.0 - error_affilation) * 100, 2)), file=f)
    # print('the final missclassed list',missclass_list_affilation, file=f)

################################################################################

if __name__ == '__main__':
    main()


