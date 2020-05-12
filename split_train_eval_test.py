'''
use the Stratified sampling to sample data. For every circle, the first 350 images are selected as training set, then there have 50 images gap;
Then 150 images are selected as testing set, finally 50 images are thrown away.
For the training set, the sampling set are selected with proportion 8:2, also Stratified sampling.
So for one circle (600 images), the whole proportion is train:eval:gap:test:gap = 280:70:50:150:50
The sampled data is saved in the 'data' document
PS: if you don't need/have the angle information, you need the delete all the code with keywords 'angle'
'''

import os, random, shutil
from sklearn.utils import shuffle
import numpy as np

def sample_images(fileDir, sample_num):
    '''
    randomly sample images
    '''
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, sample_num)
    return sample


def generate_path(root_path):

    path = []
    for root, dirs, files in os.walk(root_path, topdown=True):
        for name in dirs:
            path.append(os.path.join(root, name))
    return path


def calculate_imgs(fileDir):
    '''
    计算文件夹内文件的总数
    '''
    imgs_num = 0
    for root, dirs, files in os.walk(fileDir):  # 遍历统计
        for each in files:
            imgs_num += 1
    return imgs_num


def renew_doc():
    '''
    delete the old data and setup the new document for the new data
    '''
    try:
        if os.listdir("data/dataset-train/"):
            shutil.rmtree("data/dataset-train")
            os.mkdir("data/dataset-train")
            os.mkdir("data/dataset-train/bicycle-lane")
            os.mkdir("data/dataset-train/bicycle-lane-and-pedestrian")
            os.mkdir("data/dataset-train/car-lane")
            os.mkdir("data/dataset-train/pedestrian")
    except:
        pass

    try:
        if os.listdir("data/dataset-evaluation/"):
            shutil.rmtree("data/dataset-evaluation")
            os.mkdir("data/dataset-evaluation")
            os.mkdir("data/dataset-evaluation/bicycle-lane")
            os.mkdir("data/dataset-evaluation/bicycle-lane-and-pedestrian")
            os.mkdir("data/dataset-evaluation/car-lane")
            os.mkdir("data/dataset-evaluation/pedestrian")
    except:
        pass

    try:
        if os.listdir("data/dataset-test/"):
            shutil.rmtree("data/dataset-test")
            os.mkdir("data/dataset-test")
            os.mkdir("data/dataset-test/bicycle-lane")
            os.mkdir("data/dataset-test/bicycle-lane-and-pedestrian")
            os.mkdir("data/dataset-test/car-lane")
            os.mkdir("data/dataset-test/pedestrian")
    except:
        pass

    # try:
    #     if os.listdir("data/dataset-angle/"):
    #         shutil.rmtree("data/dataset-angle")
    #         os.mkdir("data/dataset-angle")
    #         os.mkdir("data/dataset-angle/dataset-test")
    #         os.mkdir("data/dataset-angle/dataset-train")
    # except:
    #     pass


def main():
    renew_doc()
    class_paths = generate_path("/mrtstorage/users/pan/dataset_v2")  # The original data path
    sample_paths_train = generate_path("data/dataset-train")  # sample的文件夹
    sample_paths_evaluation = generate_path("data/dataset-evaluation")
    sample_paths_test = generate_path("data/dataset-test")

    # angle_date = ('18.07-1', '18.07-2', '18.07-3','30.07-1', '30.07-2', '12.11-1', '12.11-2', '14.11-1','14.11-2')
    # angle_sum = []
    # for i in range(len(angle_date)):
    #     data = np.genfromtxt(
    #         '/media/pan/Extreme SSD/mrt-storage/dataset_rgb_depth/angle_sum_'+ angle_date[i] +'.txt',
    #         skip_header=1,
    #         dtype=None,
    #         delimiter=' ')
    #     data = data.tolist()
    #     angle_sum.append(data)

    sample = []
    i = 0
    total_sample_num_train = 0
    total_sample_num_eval = 0
    total_sample_num_test = 0

    classes = ['pedestrian','car-lane',"bicycle-lane-and-pedestrian","bicycle-lane"]

    for class_path in class_paths:  # 对每个分类文件夹分别进行操作
        if i >= 4:
            break

        second_paths = generate_path(class_path)
        second_paths.sort()
        images_path_whole = []
        if len(second_paths) != 0:
            for second_path in second_paths:  # read the path of each images
                images_path = [os.path.join(second_path, f) for f in os.listdir(second_path)]
                images_path.sort()#produce random path,but not sequence
                images_path_whole.extend(images_path)
        else:
            images_path_whole = [os.path.join(class_path, f) for f in os.listdir(class_path)]
            images_path_whole.sort()  # produce random path,but not sequence


        images_side = len(images_path_whole)
        count = 0
        images_path_train = []
        images_path_test = []
        #split the training and testing dataset with the porproty of 7:1:3:1
        for num in range(images_side):
            count += 1
            if count <= 350:
                images_path_train.append(images_path_whole[num])
            elif count > 400 and count <= 550:
                images_path_test.append(images_path_whole[num])
            elif count > 600:
                count = 0

        #extract the corresponding angle information

        # f_test = open("/home/pan/master-thesis-in-mrt/data/dataset-angle/dataset-test/" + classes[i] + ".txt", 'a+')
        # angle_test_data = []
        # for k in range(len(images_path_test)):
        #     path_tmp = images_path_test[k]
        #     path_date = path_tmp[-19:-12]
        #     path_num = int(path_tmp[-9:-4])
        #     if path_date not in angle_date:
        #         angle = 0
        #     else:
        #         a = angle_date.index(path_date)
        #         angle = angle_sum[a][path_num]
        #     angle_test_data.append(angle)
        #     print(angle,file = f_test)

        sample_num_train = len(images_path_train)//50   # here you can select the proportion of the sampling
        sample_num_test = len(images_path_test) //100   # the percentage of sampling


        sample_train = random.sample(images_path_train, sample_num_train)
        sample_test = random.sample(images_path_test,sample_num_test)
        #sample_test = images_path_test
        #randomly split train and evaluation
        sample_train = shuffle(sample_train)
        train_eval_split = 0.2                         #the percentage of randomly split train and eval
        sample_num_eval = int(train_eval_split*len(sample_train))
        sample_num_train -= sample_num_eval
        sample_eval = sample_train[:sample_num_eval]
        sample_train = sample_train[sample_num_eval:]
        print('sample %d images as training data'%(len(sample_train)))
        print('sample %d images as evaluation data' % (len(sample_eval)))
        print('sample %d images as testing data'%(len(sample_test)))

        # 复制并命名文件到新文件夹
        num_train = 0
        num_eval = 0
        num_test = 0
        for path in sample_train:  # path is the whole path
            shutil.copy(path, sample_paths_train[i] + '/')
            old_file_name = os.path.join(os.path.abspath(sample_paths_train[i]) + '/', path[-11:])
            new_file_name = os.path.join(os.path.abspath(sample_paths_train[i]) + '/',
                                         format(str(num_train), '0>5s') + '.png')
            os.rename(old_file_name, new_file_name)
            num_train += 1
        for path in sample_eval:
            shutil.copy(path, sample_paths_evaluation[i] + '/')
            old_file_name = os.path.join(os.path.abspath(sample_paths_evaluation[i]) + '/',path[-11:])
            new_file_name = os.path.join(os.path.abspath(sample_paths_evaluation[i]) + '/',
                                         format(str(num_eval), '0>5s') + '.png')
            os.rename(old_file_name,new_file_name)
            num_eval += 1
        for path in sample_test:
            shutil.copy(path, sample_paths_test[i] + '/')
            old_file_name = os.path.join(os.path.abspath(sample_paths_test[i]) + '/',path[-11:])
            new_file_name = os.path.join(os.path.abspath(sample_paths_test[i]) + '/',
                                         format(str(num_test), '0>5s') + '.png')
            os.rename(old_file_name,new_file_name)
            num_test += 1


        total_sample_num_train += num_train
        total_sample_num_eval += num_eval
        total_sample_num_test += num_test
        i += 1
    print('totally sample %d images as training data'%(total_sample_num_train))
    print('totally sample %d images as evaluation data' % (total_sample_num_eval))
    print('totally sample %d images as testing data' % (total_sample_num_test))

if __name__ == '__main__':
    main()
