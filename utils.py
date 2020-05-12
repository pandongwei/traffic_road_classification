#!/usr/bin/env python3

import os
import numpy as np
import cv2
import params
#import gist


show_additional_process_information = False
show_images_as_they_are_loaded = False

##################################################################################
'''
function about calculating the program duration time
'''
def get_elapsed_time(start):
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 1))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 2))
    return time_str


def print_duration(start):  #计算这部分程序所花时间
    time = get_elapsed_time(start)
    print(("Took {}".format(format_time(time))))

################################################################################

#seperate the training set and testing set with different strategy
def split_train_test_set(imgs_data,test_size=0.3):
    '''
    用分层分离的方法把数据分成训练集和测试集，注意不是随机分类
    比如每十张图片，前七张为训练集，后三张为测试集
    '''
    train_imgs_data = []
    test_imgs_data = []
    num_imgs = len(imgs_data)
    num_ten_imgs = num_imgs//10
    try:
        for i in range(num_ten_imgs+1):
            k = 0
            for j in range(10*i,10*i+10):
                k+=1
                if k<=10*(1-test_size):
                    train_imgs_data.append(imgs_data[j])
                else:
                    test_imgs_data.append(imgs_data[j])
    except:
        return train_imgs_data,test_imgs_data
    return train_imgs_data,test_imgs_data

################################################################################

class ImageData_multipart(object):
    '''
    use this class when need to deal with multi class image
    many function about multipart
    '''
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.class_number = None
        self.img_32parts = np.array([])

        #self.hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)  # default is 64 x 128
        self.hog = cv2.HOGDescriptor()
        self.bow_descriptor = np.array([])
        self.hog_descriptor = np.array([])
        self.gist_descriptor = np.array([])
        self.fhog_descriptor = np.array([])


        self.image_upper = np.array([])
        self.image_wide_middle = np.array([])
        self.image_wide_both_side = np.array([])
        self.image_narrow_middle = np.array([])
        self.image_narrow_both_side = np.array([])

        self.hog_descriptor_upper = np.array([])
        self.hog_descriptor_wide_middle = np.array([])
        self.hog_descriptor_wide_both_side = np.array([])
        self.hog_descriptor_narrow_middle = np.array([])
        self.hog_descriptor_narrow_both_side = np.array([])


        self.test_prob_upper = np.array([5, 1])
        self.test_prob_wide_middle = np.array([5, 1])
        self.test_prob_wide_both_side = np.array([5, 1])
        self.test_prob_narrow_middle = np.array([5, 1])
        self.test_prob_narrow_both_side = np.array([5, 1])

    def resize(self):
        self.img = cv2.resize(self.img,(224,224),interpolation=cv2.INTER_AREA)

    def set_class(self, class_name,vector):
        self.class_name = class_name
        self.class_number = get_class_number(self.class_name,vector)
        if show_additional_process_information:
            print("class name : ", class_name, " - ", self.class_number)

    def compute_hog_descriptor(self):

        # generate the HOG descriptors for a given image

        img_hog = cv2.resize(self.img,
                             (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]),
                             interpolation=cv2.INTER_AREA)

        self.hog_descriptor = self.hog.compute(img_hog)

        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if show_additional_process_information:
            print("HOG descriptor computed - dimension: ", self.hog_descriptor.shape)

    '''  
    # FHOG feature cannot be extracted in python3 environment, only in python2 with specific package (cannot remember which one)
    def compute_fhog_descriptor(self):

        # generate the HOG descriptors for a given image

        img_hog = cv2.resize(self.img,
                             (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]),
                             interpolation=cv2.INTER_AREA)

        self.fhog_descriptor = pyhog.features_pedro(img_hog, 30)

        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if show_additional_process_information:
            print("HOG descriptor computed - dimension: ", self.hog_descriptor.shape)
    '''
    def compute_bow_descriptor(self):

        # generate the feature descriptors for a given image

        self.bow_descriptor = params.DETECTOR.detectAndCompute(self.img, None)[1]

        if self.bow_descriptor is None:
            self.bow_descriptor = np.array([])

        if show_additional_process_information:
            print("# feature descriptors computed - ", len(self.bow_descriptor))

    '''
    # this function
    def compute_gist_descriptor(self):

        # generate the gist feature descriptors for a given image

        self.gist_descriptor = gist.extract(self.img, nblocks=4, orientations_per_scale=(2,2))

        if self.gist_descriptor is None:
            self.gist_descriptor = np.array([])

        if show_additional_process_information:
            print("# gist feature descriptors computed - ", len(self.gist_descriptor))
    '''

    def compute_hog_descriptor_multipart(self):

        # generate the HOG descriptors for 5 parts
        imgs_part = [self.image_upper,self.image_wide_middle,self.image_wide_both_side,
                  self.image_narrow_middle,self.image_narrow_both_side]
        hog_descriptors = [self.hog_descriptor_upper,self.hog_descriptor_wide_middle,self.hog_descriptor_wide_both_side,
                           self.hog_descriptor_narrow_middle,self.hog_descriptor_narrow_both_side]
        i = 0
        for img_part,hog_descriptor in zip(imgs_part,hog_descriptors):
            img_hog = cv2.resize(img_part, (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]), interpolation = cv2.INTER_AREA)

            hog_descriptor = self.hog.compute(img_hog)

            if hog_descriptor is None:
                hog_descriptor = np.array([])

            if show_additional_process_information:
                print("HOG descriptor computed - dimension: ", hog_descriptor.shape)
            if i == 0:
                self.hog_descriptor_upper = hog_descriptor
            elif i == 1:
                self.hog_descriptor_wide_middle = hog_descriptor
            elif i == 2:
                self.hog_descriptor_wide_both_side = hog_descriptor
            elif i == 3:
                self.hog_descriptor_narrow_middle = hog_descriptor
            elif i == 4:
                self.hog_descriptor_narrow_both_side = hog_descriptor
            i+= 1


    def generate_bow_hist(self, dictionary):
        self.bow_histogram = np.zeros((len(dictionary), 1))  #建立空的数组

        # generate the bow histogram of feature occurance from descriptors
            # FLANN matcher with SIFT/SURF needs descriptors to be type32
        matches = params.MATCHER.match(np.float32(self.bow_descriptor), dictionary)
            #另一种params.MATCHER.knnMatch()，returns k best matches where k is specified by the user.

        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram (known as hard assignment)
            self.bow_histogram[match.trainIdx] += 1

        # Important! - normalize the histogram to L1 to remove bias for number
        # of descriptors per image or class (could use L2?)

        self.bow_histogram = cv2.normalize(self.bow_histogram, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)


    def segment_imagedata_into_5_parts(self):

        roi_corners_upper = np.array([[(0, 0), (1024, 0), (1024, 160), (0, 160)]], dtype=np.int32)
        self.image_upper = ImageData_multipart.split_image(self,roi_corners_upper)

        roi_corners_wide_middle = np.array([[(400, 160), (0, 400), (0, 540), (1024, 540), (1024, 400), (624, 160)]],dtype=np.int32)
        self.image_wide_middle = ImageData_multipart.split_image(self,roi_corners_wide_middle)

        roi_corners_wide_both_side_1 = np.array([[(0, 160), (0, 400), (400, 160)]], dtype=np.int32)
        roi_corners_wide_both_side_2 = np.array([[(1024, 160), (1024, 400), (600, 160)]], dtype=np.int32)
        self.image_wide_both_side = ImageData_multipart.split_image_both_side(self, roi_corners_wide_both_side_1,roi_corners_wide_both_side_2)

        roi_corners_narrow_middle = np.array([[(460, 160), (150, 540), (874, 540), (564, 160)]], dtype=np.int32)
        self.image_narrow_middle = ImageData_multipart.split_image(self, roi_corners_narrow_middle)

        roi_corners_narrow_both_side_1 = np.array([[(0, 160), (0, 540), (150, 540), (460, 160)]], dtype=np.int32)
        roi_corners_narrow_both_side_2 = np.array([[(564, 160), (874, 540), (1024, 540), (1024, 160)]], dtype=np.int32)
        self.image_narrow_both_side = ImageData_multipart.split_image_both_side(self, roi_corners_narrow_both_side_1, roi_corners_narrow_both_side_2)

    def split_image(self,roi_corners_1):
        origin_image = self.img
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        channel_count = self.img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, roi_corners_1, ignore_mask_color)
        # apply the mask
        masked_image = cv2.bitwise_and(origin_image, mask)

        return masked_image

    def split_image_both_side(self,roi_corners_1,roi_corners_2):
        origin_image = self.img
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        channel_count = self.img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, roi_corners_1, ignore_mask_color)
        cv2.fillPoly(mask, roi_corners_2, ignore_mask_color)
        # apply the mask
        masked_image = cv2.bitwise_and(origin_image, mask)

        return masked_image

    def segment_imagedata_into_5_parts_for_test(self):

        roi_corners_upper = np.array([[(0, 0), (1024, 0), (1024, 160), (0, 160)]], dtype=np.int32)
        self.image_upper = ImageData_multipart.split_image(self, roi_corners_upper)
        cv2.imshow('1', self.image_upper)
        cv2.waitKey(0)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
        roi_corners_wide_middle = np.array([[(400, 160), (0, 400), (0, 540), (1024, 540), (1024, 400), (624, 160)]],
                                           dtype=np.int32)
        self.image_wide_middle = ImageData_multipart.split_image(self, roi_corners_wide_middle)
        cv2.imshow('1', self.image_wide_middle)
        cv2.waitKey(0)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
        roi_corners_wide_both_side_1 = np.array([[(0, 160), (0, 400), (400, 160)]], dtype=np.int32)
        roi_corners_wide_both_side_2 = np.array([[(1024, 160), (1024, 400), (600, 160)]], dtype=np.int32)
        self.image_wide_both_side = ImageData_multipart.split_image_both_side(self, roi_corners_wide_both_side_1,
                                                                    roi_corners_wide_both_side_2)
        cv2.imshow('1', self.image_wide_both_side)
        cv2.waitKey(0)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
        roi_corners_narrow_middle = np.array([[(460, 160), (150, 540), (874, 540), (564, 160)]], dtype=np.int32)
        self.image_narrow_middle = ImageData_multipart.split_image(self, roi_corners_narrow_middle)
        cv2.imshow('1', self.image_narrow_middle)
        cv2.waitKey(0)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()
        roi_corners_narrow_both_side_1 = np.array([[(0, 160), (0, 540), (150, 540), (460, 160)]], dtype=np.int32)
        roi_corners_narrow_both_side_2 = np.array([[(564, 160), (874, 540), (1024, 540), (1024, 160)]],
                                                  dtype=np.int32)
        self.image_narrow_both_side = ImageData_multipart.split_image_both_side(self, roi_corners_narrow_both_side_1,
                                                                      roi_corners_narrow_both_side_2)
        cv2.imshow('1', self.image_narrow_both_side)
        cv2.waitKey(0)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def split_img_into_32parts(self):
        img_hog = cv2.resize(self.img,
                             (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]),
                             interpolation=cv2.INTER_AREA)
        img_part_width = params.DATA_WINDOW_SIZE[0]//8
        img_part_height = params.DATA_WINDOW_SIZE[1]//4
        list_32parts = []
        for i in range(4):
            for j in range(8):
                img_part = img_hog[i*img_part_height:(i+1)*img_part_height,
                           j*img_part_width:(j+1)*img_part_width]
                list_32parts.append(img_part)
        self.img_32parts = np.array(list_32parts)

    def compute_hog_descriptor_32parts(self):

        # generate the HOG descriptors for a given image
        hog_32parts = []
        for i in range(32):

            hog_32parts.append(self.hog.compute(self.img_32parts[i]))


        self.hog_descriptor = np.array(hog_32parts)

        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if show_additional_process_information:
            print("HOG descriptor computed - dimension: ", self.hog_descriptor.shape)

    def compute_hog_descriptor_3parts(self):

        # generate the HOG descriptors for a given image
        hog_3parts = []
        hog_upper = []
        hog_both_side = []
        hog_middle = []
        for i in range(32):
            if i < 8:
                hog_upper.append(self.hog.compute(self.img_32parts[i]))
            elif 8<= i <= 10 or 13<= i <=17 or 22<= i <=23:
                hog_both_side.append(self.hog.compute(self.img_32parts[i]))
            else:
                hog_middle.append(self.hog.compute(self.img_32parts[i]))

        hog_3parts.append(np.mean(hog_upper,axis = 0))
        hog_3parts.append(np.mean(hog_middle,axis = 0))
        hog_3parts.append(np.mean(hog_both_side,axis = 0))
        self.hog_descriptor = np.array(hog_3parts)

        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if show_additional_process_information:
            print("HOG descriptor computed - dimension: ", self.hog_descriptor.shape)


################################################################################

def generate_path(root_path):
    '''
    生成目录下所有文件夹的路径
    '''
    path = []
    class_names = []
    for root, dirs, files in os.walk(root_path, topdown=True):
        for name in dirs:
            path.append(os.path.join(root, name))
            class_names.append(name)
    path.sort()
    class_names.sort()
    return path, class_names


def load_images(paths, class_names,vector = False):
    # load image data from specified paths
    #vector = True,则class label以向量形式输出

    imgs_data = []  # type: list[ImageData]

    # for each specified path and corresponding class_name and required number
    # of samples - add them to the data set
    for path, class_name in zip(paths, class_names):
        load_image_path(path, class_name, imgs_data,vector)

    return imgs_data


def load_image_path(path, class_name, imgs_data,vector = False):

    #vector = True,则class label以向量形式输出

    # reads all the images in a given folder path and returns the results
    images_path = [os.path.join(path, f) for f in os.listdir(path)]  #path加上图片的文件名
    images_path.sort()
    images = []
    for image_path in images_path:

        if (('.png' in image_path) or ('.jpg' in image_path)):
            img = cv2.imread(image_path)
            images.append(img)
        else:
            if show_additional_process_information:
                print("skipping non PNG/JPG file - ", image_path)

    img_count = len(imgs_data)
    for img in images:

        if (show_images_as_they_are_loaded):
            cv2.imshow("example", img)
            cv2.waitKey(5)

        img_data = ImageData_multipart(img)
        img_data.set_class(class_name,vector)
        imgs_data.insert(img_count, img_data)
        img_count += 1

    return imgs_data

################################################################################

# return the global set of bow histograms for the data set of images

# stack array of items as basic Pyton data manipulation

def stack_array(arr):
    stacked_arr = np.array([])
    for item in arr:
        # Only stack if it is not empty
        if len(item) > 0:
            if len(stacked_arr) == 0:
                stacked_arr = np.array(item)
            else:
                stacked_arr = np.vstack((stacked_arr, item))  #Stack arrays in sequence vertically (row wise).
    return stacked_arr

def get_bow_histograms(imgs_data):

    samples = stack_array([[img_data.bow_histogram] for img_data in imgs_data])
    return np.float32(samples)

################################################################################

# return the global set of hog descriptors for the data set of images

def get_hog_descriptors(imgs_data):

    samples = stack_array([[img_data.hog_descriptor] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_upper(imgs_data):

    samples = stack_array([[img_data.hog_descriptor_upper] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_wide_middle(imgs_data):

    samples = stack_array([[img_data.hog_descriptor_wide_middle] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_wide_both_side(imgs_data):

    samples = stack_array([[img_data.hog_descriptor_wide_both_side] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_narrow_middle(imgs_data):

    samples = stack_array([[img_data.hog_descriptor_narrow_middle] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_narrow_both_side(imgs_data):

    samples = stack_array([[img_data.hog_descriptor_narrow_both_side] for img_data in imgs_data])
    return np.float32(samples)

def get_hog_descriptors_32parts(imgs_data,img_part_i):

    samples = stack_array([[img_data.hog_descriptor[img_part_i]] for img_data in imgs_data])
    return np.float32(samples)


def get_gist_descriptors(imgs_data):

    samples = stack_array([[img_data.gist_descriptor] for img_data in imgs_data])
    return np.float32(samples)
################################################################################

def get_class_number(class_name,vector):
    if vector:
        return params.DATA_CLASS_NAMES_in_vector.get(class_name)
    else:
        return params.DATA_CLASS_NAMES.get(class_name)

def get_class_name(class_code):
    for name, code in params.DATA_CLASS_NAMES.items():
        if code == class_code:
            return name

# return global the set of numerical class labels for the data set of images
def get_class_labels(imgs_data):
    class_labels = [img_data.class_number for img_data in imgs_data]
    return np.int32(class_labels)
##################################################################################
'''
function about calculating the error in training and testing
'''
def errorCalculation(class_labels,output):
    '''
    calculate the error between the prediction class and the true class
    '''
    res = 0
    for i in range(len(class_labels)):
        if class_labels[i] != output[i]:
            res +=1
        i+=1   #加不加都可以
    error = res/float(len(output))
    return error

def errorCalculation_return_missclassList(class_labels,output):
    res = 0
    missclass_list = []
    for i in range(len(class_labels)):
        if class_labels[i] != output[i]:
            res +=1
            missclass_list.append(i)
        i+=1
    error = res/float(len(output))
    return error,missclass_list


def errorCalculation_affilation_return_missclassList(class_labels,output):
    for i in range(len(output)):
        if output[i] == 1:
            output[i] = 0
        if class_labels[i] == 1:
            class_labels[i] = 0
    error, missclass_list = errorCalculation_return_missclassList(class_labels,output)
    return error,missclass_list,output


def show_missclass_image(test_imgs_data,class_labels,output,missclass_list,prediction,show_the_image = False):
    DATA_CLASS_NAMES = {
        0:"bicycle-lane",
        1:"bicycle-lane-and-pedestrian",
        2:"car-lane",
        3:"pedestrian"
    }
    bl_corner = (10, 530)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(missclass_list)):
        print('the label of image {0} is: {1}'.format(missclass_list[i],DATA_CLASS_NAMES[class_labels[missclass_list[i]]]))
        print('the image is classified to: ', DATA_CLASS_NAMES[output[missclass_list[i]]])
        print(prediction[missclass_list[i]])  # show the probability of the missclassified image
        if show_the_image:
            img = test_imgs_data[missclass_list[i]].img
            cv2.putText(img, "Labels: " + str(DATA_CLASS_NAMES[class_labels[missclass_list[i]]]), (bl_corner[0], bl_corner[1] - 20), font, 1, (0, 0, 255), 2)
            cv2.putText(img, "Classified: " + str(DATA_CLASS_NAMES[output[missclass_list[i]]]), bl_corner, font, 1, (0, 255, 0), 2)
            cv2.imshow("classification_visualization", img)
            if cv2.waitKey(0) or 0xFF == ord('q'):
                cv2.destroyAllWindows()
        print('the label of image {0} is: {1}'.format(missclass_list[i],DATA_CLASS_NAMES[class_labels[missclass_list[i]]]))
        print('the image is classified to: ', DATA_CLASS_NAMES[output[missclass_list[i]]])
