'''
train the model MobileNetV3 with LSTM layer to extract the spatial information in one image
'''

import os
import json
import pandas as pd

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint,LearningRateScheduler
from datetime import datetime
from utils import *
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import numpy as np
import cv2
import math
from PIL import Image
from sklearn.utils import shuffle
from kalman_filter import *

def cosine_decay(epoch):
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    initial_lrate = cfg['learning_rate']
    epochs = cfg['epochs']
    lr = 0.5*(1+math.cos(math.pi*epoch/epochs))*initial_lrate
    return lr

def generate(batch, shape, ptrain, pval):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
        ptrain: train dir.
        pval: eval dir.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(rescale=1. / 255,
                                  vertical_flip=True,
                                  zca_whitening=True,
                                  zoom_range=0.2)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        shuffle=True,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=batch,
        shuffle=False,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1
    '''
    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        batch_size=count2,
        shuffle=True,
        class_mode='categorical')
    '''
    return train_generator, validation_generator, count1, count2


def train():
    ## no need when using watcher
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## todo
    config_keras = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))
    config_keras.gpu_options.allow_growth = True
    session = tf.Session(config=config_keras)
    K.set_session(session)

    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])
    learningRate = cfg['learning_rate']

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    from model.mobilenet_v3_large_LSTM import MobileNetV3_Large_LSTM
    model = MobileNetV3_Large_LSTM(shape, n_class).build()

    pre_weights = cfg['weights']
    if pre_weights and os.path.exists(pre_weights):
        model.load_weights(pre_weights, by_name=True, skip_mismatch=True)

    opt = Adam(lr=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    train_generator, validation_generator, count1, count2 = generate(batch, shape[:2], cfg['train_dir'], cfg['eval_dir'])

    #callback function
    #earlystop = EarlyStopping(monitor='val_acc', patience=3, verbose=0, mode='auto')
    log_dir = "train_result/lightweight_network/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
    checkpoint = ModelCheckpoint(log_dir + '/weights_best.h5',
                                 monitor='val_accuracy', save_weights_only=True, save_best_only=False,mode='auto')
    #lrate = LearningRateScheduler(cosine_decay)

    hist = model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2//batch,
        epochs=cfg['epochs'],
        shuffle=True,
        callbacks=[tensorboard_callback,checkpoint])

    model.save_weights(os.path.join(log_dir, '{}_weights_final.h5'.format(cfg['model'])))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(log_dir, 'hist.csv'), encoding='utf-8', index=False)

    # test the result
    test_dir = cfg['test_dir']
    print('loading image.....')
    start = cv2.getTickCount()
    path, class_names = generate_path(test_dir)
    test_imgs_data = load_images(path, class_names)
    print_duration(start)
    class_labels = np.int32([img_data.class_number for img_data in test_imgs_data])
    # samples = stack_array([[cv2.cvtColor(img_data.img,cv2.COLOR_BGR2RGB)] for img_data in test_imgs_data])
    samples = []
    for img_data in test_imgs_data:
        samples.append(cv2.cvtColor(img_data.img, cv2.COLOR_BGR2RGB))
    samples = np.asanyarray(samples)
    print_duration(start)
    print('finish loading image')
    samples = samples / 255.
    predictions = model.predict(samples, batch_size=100)
    print_duration(start)
    f1 = open("result.txt", 'a+')
    for i in range(len(predictions)):
        print(predictions[i],file = f1,end=' ')
    output = predictions.argmax(1)
    num_correct = sum(np.array(output) == np.array(class_labels))
    print(num_correct)
    acc = num_correct / len(class_labels)
    print('test result:  ', acc)

    #combine kalman filter
    predict_prob = predictions.T
    output_kalman = []
    x_state = KalmanFilter()
    for i in range(len(predict_prob[0])):
        #kalman part
        x_state.predict()
        x_state.correct(predict_prob[:,i])

        x_combination = x_state.x_e
        output_kalman.append(np.argmax(x_combination))

    f = open("result_summery.txt", 'a+')
    print('the model is MobilenetV3',file = f)
    error_kalman, missclass_list_kalman = errorCalculation_return_missclassList(class_labels, output_kalman)
    print("after combining kalman filter the classifier got {}% of the testing examples correct!".format(round((1.0 - error_kalman) * 100, 2)))
    #write the result in the txt
    print('test accuracy without kalman 4 classes: {}%'.format(round(acc*100, 2)), file=f)
    print('test accuracy with kalman 4 classes: {}%'.format(round((1.0 - error_kalman) * 100, 2)), file=f)
    #print(missclass_list_kalman,file = f)

    #add affilation
    error_affilation, missclass_list_affilation = errorCalculation_affilation_return_missclassList(class_labels, output_kalman)
    print("after affilation to 3 classes the classifier got {}% of the testing examples correct!".format(round((1.0 - error_affilation) * 100, 2)))
    print('test accuracy with kalman 3 classes: {}%'.format(round((1.0 - error_affilation) * 100, 2)), file=f)
    print('the final missclassed list',missclass_list_affilation, file=f)


if __name__ == '__main__':
    train()
