import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
import cv2
import numpy as np
import os
from random import shuffle

Train_Path = r'D:\Dog_CatDatabase\train'
Test_Path = r'D:\Dog_CatDatabase\test_1500~3000'
IMG_SIZE = 50
LR = 1e-3

PROJECT_NAME = 'dogsVScats-{}-{}.model'.format('6conv',LR)

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'dog': return [1,0]
    elif word_label == 'cat': return [0,1]

def create_train_data():
    trainimg_data = []
    for img in tqdm(os.listdir(Train_Path)):
        label = label_img(img)
        path = os.path.join(Train_Path,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        trainimg_data.append([np.array(img),np.array(label)])
    shuffle(trainimg_data)
    np.save('train_data.npy',trainimg_data)
    return trainimg_data


train_data = create_train_data()
# 如果已经有训练完的可以载入
# train_data = np.load('train_data.npy')

tf.reset_default_graph

convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1], name='input')

convnet = conv_2d(convnet, nb_filter=32, filter_size=2, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, n_units=1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.mata'.format(PROJECT_NAME)):
    model.load(PROJECT_NAME)
    print('project loading!')

train = train_data[:-500]
check = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
Y = [i[1] for i in train]

check_x = np.array([i[0] for i in check]).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
check_y = [i[1] for i in check]

model.fit({'input': X},{'targets': Y}, validation_set=({'input': check_x},{'targets': check_y}), snapshot_step= 500, show_metric=True, run_id=PROJECT_NAME)

# 终端命令，tensorboard --logdir=foo:D:\NeuralNetwork\log
# CD到 log 文件夹下：tensorboard --logdir logs
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_Path)):
        path = os.path.join(Test_Path,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])

    np.save('test_daa.npy',testing_data)
    return testing_data
test_data = process_test_data()  #没有test_data
# test_data = np.load('test_data.npy')  有test_data

fig = plt.figure()

for num, data in enumerate(test_data[:20]):
    img_num = data[1]
    img_data = data [0]

    y = fig.add_subplot(4,5,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1: str_label='Cat'
    else: str_label = 'Dog'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()