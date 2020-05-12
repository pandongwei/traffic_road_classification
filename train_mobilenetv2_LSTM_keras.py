"""
Train the MobileNet V2 + LSTM, to extract the spatial information in one image
Here i use keras model because in the training i can easily set the trainable layers.
Strategy:
After loading the pretrained model, firstly set only the lstm part and fully connection layers trainable. After several epoch set all the layers trainable, and continue training
"""
import os
import pandas as pd
import json
#from mobilenet_v2 import MobileNetv2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
from keras.layers import Conv2D, Reshape, Activation,GlobalAveragePooling2D,Dense,CuDNNLSTM,TimeDistributed
from keras.models import Model,model_from_json
from datetime import datetime
from keras.backend import tensorflow_backend as K
import tensorflow as tf

from keras.applications.mobilenet_v2 import MobileNetV2


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
                                  zoom_range=0.2
                                  )

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
        shuffle=True,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2


def fine_tune(num_classes, weights, model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights(weights)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model


def train():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## todo
    config_keras = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    config_keras.gpu_options.allow_growth = True
    session = tf.Session(config=config_keras)
    K.set_session(session)

    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    batch = int(cfg['batch'])
    epochs = int(cfg['epochs'])
    num_classes = int(cfg['class_number'])
    shape = (int(cfg['height']), int(cfg['width']), 3)
    learning_rate = cfg['learning_rate']
    pre_weights = cfg['weights']

    train_generator, validation_generator, count1, count2 = generate(batch, shape[:2], cfg['train_dir'], cfg['eval_dir'])
    '''
    #use the pretrain model, finetune
    base_model = MobileNetV2(input_shape=shape,weights = 'imagenet',include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs= base_model.input,outputs = predictions)
    '''
    # save the model
    #model_json = model.to_json()
    #with open('model/v2_model_architecture_finetune_4.json', 'w') as f:
    #    f.write(model_json)

    # load the weight
    with open('model/mobilenetv2_keras_finetune_model.json', 'r') as f:
        model = model_from_json(f.read())
    if pre_weights and os.path.exists(pre_weights):
        model.load_weights(pre_weights, by_name=True,skip_mismatch=True)



    opt = Adam(lr=learning_rate)
    log_dir = "train_result/lightweight_network/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=420)
    checkpoint = ModelCheckpoint(log_dir + '/weights_pre.h5',
                                 monitor='val_accuracy', save_weights_only=False,save_best_only=False)
    #earlystop = EarlyStopping(monitor='val_acc', patience=3, verbose=0, mode='auto')

    # add two lstm layers and two fully connection layers
    x = model.get_layer('block_16_project_BN').output
    x = TimeDistributed(CuDNNLSTM(320))(x)
    x = CuDNNLSTM(320)(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(4,activation='softmax')(x)
    model = Model(model.input,x)

    # set all the layers are trainable
    for layer in model.layers[:152]:
        layer.trainable = True
    for layer in model.layers[152:]:
        layer.trainable = True


    #重新编译
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #tf.keras.Model.save(model,os.path.join('logs/fit/20191223-124218/', 'model1_final.h5'))
    model.summary()
    hist = model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=[tensorboard_callback,checkpoint])

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(log_dir+'/hist.csv', encoding='utf-8', index=False)
    model.save_weights(log_dir+'/pretrain_final_weights.h5')
    model.save(os.path.join(log_dir, 'model_final.h5'))

if __name__ == '__main__':
    train()