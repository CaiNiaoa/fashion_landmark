# -*- coding: utf-8 -*-

# The Python standard libraries
import os

# The third-party libraries
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# My libraries
import fashion_landmark as kpl_1_0

DATASET_PATH = '../dataset/'
NPY_MODEL_PATH = 'pretrain_model/VGG_imagenet.npy'
CKPT_MODEL_PATH = './model/' # 模型保存的地址
CKPT_MODEL_NAME = 'key_points_location.ckpt' # 模型保存的名字
LOG_PATH = './log/' # 保存TensorBoard日志的地址，用来查看计算图和显示程序中可视化的参数

BATCH_SIZE = 8 # 一个batch的大小
NUM_BATCH = 80000 # batch数目
LEARNING_RATE = 0.01 # 学习率

columns_buf = [ # 结果文档的各列名称

'image_id', 'image_category',
'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'

]

def draw_key_points(img, points):
    for i in range(int(len(points) / 2)):
        x, y = points[i*2], points[i*2+1]
        cv2.circle(img, (x, y), 3, (10, 10, 180), 3)
    cv2.imshow('key_points', img)
    cv2.waitKey(0)

def test(dataset):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x-input")

    pre = kpl_1_0.model(x, num_stage=1, train_flag=False)

    # 定义保存网络参数的Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CKPT_MODEL_PATH) # 加载checkpoint，
        if ckpt:
            print ('Load the parameters from ' +  ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # 如果该文件存在，则使用其中保存的网络参数初始化当前网络的参数，然后继续训练，

            for i in range(400, 430):
                img_path = os.path.join(DATASET_PATH, 'test', dataset.test_img_ids[i])
                img = cv2.imread(img_path)
                assert (img != None)
                h, w = img.shape[0], img.shape[1] # The picture's format is HWC. x is W, y is H.
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                assert (img.shape == (224, 224, 3))
                img = np.reshape(img, (1, 224, 224, 3))
                # h_k, w_k = 224.0 / h, 224.0 / w

                pos = sess.run(pre, feed_dict={x: img})
                pos = np.squeeze(pos, axis=0)
                print('Image: ' + dataset.test_img_ids[i] + '========')
                print(pos)
                print('\n')
                points = np.array(pos, np.int16)
                points[np.where(points[:] < 0)] = 0
                points[np.where(points[:] > 223)] = 223

                img = np.reshape(img, (224, 224, 3))
                draw_key_points(img, points)
        else:
            print ("No checkpoint file.")
            return


if __name__ == "__main__":
    dataset = kpl_1_0.Dataset(DATASET_PATH)
    test(dataset)