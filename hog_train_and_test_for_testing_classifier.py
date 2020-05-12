'''
This file is to test the result of different classifiers/Ensemble with HOG features, you can change the different classifier by name the 'clf'
some paths need to be assigned with TODO signs
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
def main():
    
    ############################################################################
    # load our training data set of images examples

    program_start = cv2.getTickCount()

    print("Loading images...")
    start = cv2.getTickCount()
    # the training path TODO
    paths,class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-train")
    train_imgs_data = load_images(paths, class_names)

    #paths_extra,class_names_extra = generate_path("/home/pan/master-thesis-in-mrt/data/dataset-evaluation")
    #train_imgs_data_extra = load_images(paths_extra, class_names_extra)
    #train_imgs_data.extend(train_imgs_data_extra)
    train_imgs_data = shuffle(train_imgs_data)
    print(("Loaded totally {} image(s)".format(len(train_imgs_data))))
    print_duration(start)

    ############################################################################

    print("Computing HOG descriptors...") # for each training image
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in train_imgs_data]
    print_duration(start)

    ############################################################################
    # train an SVM based on these norm_features

    print("Training Classifier...")
    start = cv2.getTickCount()

    # compile samples (i.e. visual word histograms) for each training image

    samples = get_hog_descriptors(train_imgs_data)    #提取出histogram
    #数据降维 reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    d2_train_dataset = samples.reshape((nsamples,nx*ny))

    # get class label for each training image
    class_labels = get_class_labels(train_imgs_data)
    class_labels = class_labels.ravel()

    # change the classifier TODO
    clf = BaggingClassifier(base_estimator=SVC(probability = True,gamma='scale',tol = 1e-5,C = 10), n_estimators=10, bootstrap=True, bootstrap_features=True, n_jobs=10)
    #clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1)
    #clf = GradientBoostingClassifier(n_estimators=100)
    #estimator_list = [('bagging',clf1),('rf',clf2),('gbc',clf3)]
    #clf = VotingClassifier(estimators=estimator_list,voting='hard',n_jobs=-1)
    #clf = AdaBoostClassifier(base_estimator=SVC(probability = True,tol = 1e-5,gamma='scale',C = 1),n_estimators=20)
    #clf = SVC(probability = True,tol = 1e-5,gamma='scale',C = 10)

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
    # here you need the change the path to save the result TODO
    dump(clf, 'train_result/train_whole_dataset_svm_3,7.joblib')

    #error = errorCalculation(class_labels.ravel(),result)
    print(result)

    print_duration(start)

    print(("Finished training HOG detector. {}".format(format_time(get_elapsed_time(program_start)))))

    #input("Press Enter to continue...")

    #测试部分开始
    print("Start testing ....")

    start = cv2.getTickCount()
    # the testing data path TODO
    path, class_names = generate_path("/home/pan/traffic_road_classification/data/dataset-test")

    test_imgs_data = load_images(path, class_names)
    #test_imgs_data = shuffle(test_imgs_data)
    print("testing with {} test-images".format(len(test_imgs_data)))
    print_duration(start)

    [img_data.compute_hog_descriptor() for img_data in test_imgs_data]

    samples = get_hog_descriptors(test_imgs_data)
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
    '''
    output = clf.predict(d2_test_dataset)
    error,missclass_list = errorCalculation_affilation_return_missclassList(class_labels,output)
    print("With affilation, the classifier got {}% of the testing examples correct!".format(round((1.0 - error) * 100,2)))
    print_duration(program_start)

    #output the miss classified image list
    f = open("test_result/result-without-affilation.txt", 'a+')
    print('the classifier is: ',clf,file=f)
    print('test accuracy: {}%'.format(round(result*100,2)),file=f)
    #print('test accuracy after affilation: {}%'.format(round((1.0 - error) * 100,2)),file=f)
    print(missclass_list, file = f)
    #print the missclassified image
    #show_missclass_image(test_imgs_data,class_labels,output,missclass_list,show_the_image = True)
    print(' ', file = f)
    print(' ', file = f)
    '''

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

    f = open("result_summery.txt", 'a+')
    error_kalman, missclass_list_kalman = errorCalculation_return_missclassList(class_labels, output_kalman)
    print("after combining kalman filter the classifier got {}% of the testing examples correct!".format(round((1.0 - error_kalman) * 100, 2)))
    #write the result in the txt
    print('the classifier is: ', clf, file=f)
    print('test accuracy without kalman 4 classes: {}%'.format(round(result*100, 2)), file=f)
    print('test accuracy with kalman 4 classes: {}%'.format(round((1.0 - error_kalman) * 100, 2)), file=f)
    #print(missclass_list_kalman,file = f)

    #add affilation
    error_affilation, missclass_list_affilation, _ = errorCalculation_affilation_return_missclassList(class_labels, output_kalman)
    print("after affilation to 3 classes the classifier got {}% of the testing examples correct!".format(round((1.0 - error_affilation) * 100, 2)))
    print('test accuracy with kalman 3 classes: {}%'.format(round((1.0 - error_affilation) * 100, 2)), file=f)
    print('the final missclassed list',missclass_list_affilation, file=f)

################################################################################

if __name__ == '__main__':
    main()


