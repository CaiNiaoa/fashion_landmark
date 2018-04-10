import numpy as np
import cv2
import pandas as pd
import os
import tensorflow as tf
from utils import *
import vgg16g
from network import *

DATASET_PATH = './dataset'

CKPT_MODEL_PATH = './model/'  # 模型保存的地址
CKPT_MODEL_NAME = 'key_points_location.ckpt'  # 模型保存的名字
LOG_PATH = './log/'  # 保存TensorBoard日志的地址，用来查看计算图和显示程序中可视化的参数

BATCH_SIZE = 16
LEARNING_RATE = 0.001
BATCH_NUM = 80000
NUM_STAGE = 1

#不同服饰存在的关键点标号
CATEGORY = ['blouse', 'skirt', 'outwear', 'dress','trousers']

POINT_INDEX = {'blouse': [0,1,2,3,4,5,6,9,10,11,12,13,14],
              'skirt': [15,16,17,18],
              'outwear': [0,1,3,4,5,6,7,8,9,10,11,12,13,14],
              'dress': [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18],
              'trousers': [15,16,19,20,21,22,23]}

class Dataset():
    def __init__(self,dataset_path, category):
        self.dataset_path = dataset_path
        self.category = category
        self.batch_index = 0
        self.train_image_ids, self.train_image_anno = self.load_csv('train')
        self.epoch = 0


    def load_csv(self, filename):
        '''brief:  读入训练数据以及标签 (未分类)
           input:  文件名（train / test）
           return: 分离数据集的图片ｉｄ及图片标签'''
        if filename == 'train':
            anno_path = os.path.join(DATASET_PATH,filename, 'Annotations', filename +'.csv')
        if filename== 'test':
            anno_path = os.path.join(DATASET_PATH, filename, filename + '.csv')
        print("anno_path: ",anno_path)

        data_info = load_csv(anno_path)
        print('data_info.shape: ', data_info.shape)

        #只选某一类别
        one_category_data = data_info[np.where(data_info[:, 1] == self.category)[0],:]
        print('one_category_data.shape: ',one_category_data.shape)

        # # 剔除关键点个数少的数据　分离不同数据 (此处不做剔除和数据改变)
        # if filename == 'train':
        #     trousers_data_filt = np.array([trousers for trousers in trousers_data if np.sum(trousers[2:] != '-1_-1_-1') == 7])
        #     trousers_data[np.where(trousers_data[::] == '-1_-1_-1')] = '0_0_-1'
        #     trousers_data_filt = trousers_data
        #     print('trousers_data_filt.shape: ', trousers_data_filt.shape)
        #
        # if filename == 'test':
        #     trousers_data_filt = trousers_data

        # image_ids = trousers_data_filt[:,0].copy()
        image_ids = one_category_data[:,0].copy()
        print('image_ids.shape: ', image_ids.shape)

        return image_ids, data_info



    def shuffle_train_img_ids(self):
        '''brief:打乱图像ｉｄ'''
        #self.batch_perm = np.random.permutation(np.arange(len(self.train_image_ids)))
        np.random.shuffle(self.train_image_ids)
        self.batch_index = 0

    def get_minibatch_ids(self):
        '''brief: '''
        if self.batch_index + BATCH_SIZE >= len(self.train_image_ids):
            self.shuffle_train_img_ids()
            self.epoch = self.epoch + 1

        ids = np.arange(self.batch_index, self.batch_index + BATCH_SIZE)
        self.batch_index += BATCH_SIZE

        return ids


    def get_minibatch(self, minibatch_ids):
        ''''''
        pos_labels = []
        images = []
        masks = [] #添加用于屏蔽标签为０的输出
        vis_labels = []

        for i in minibatch_ids:
            image = load_image(self.train_image_ids[i], 'train')

            train_category_axis = self.train_image_anno[np.where(self.train_image_anno[:,0] == self.train_image_ids[i])[0], 2:]
            #剔除除了７个点以外的－１
            # train_trousers_axis = train_trousers_axis[np.where(train_trousers_axis[:] != '-1_-1_-1')]

            # print('train_category_axis',train_category_axis.shape)

            #选取给定类别含有的关键点
            train_category_axis = train_category_axis[:,POINT_INDEX[self.category]]
            train_category_axis[np.where(train_category_axis[:] == '-1_-1_-1')] = '0_0_-1'

            pos = np.array([[int(int(xy.split('_')[0]) / image.shape[1] * 224), int(int(xy.split('_')[1]) / image.shape[0] * 224)] for xy in train_category_axis[0]])

            #要注意vis的维度问题[bitch_size, output_num, 3]
            vises = [int(vis.split('_')[2]) for vis in train_category_axis[0]]  #长度为output_num的list
            vises_one_hot = []
            #此处将-1,0,1转换为one_hot中0,1,2
            for vis in vises:
                vis_one_hot = np.zeros(3)
                vis_one_hot[vis + 1] = 1
                vises_one_hot.append(vis_one_hot)

            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            images.append(image)

            pos_labels.append(pos.flatten())

            pos[np.where(pos[::] != 0)] = 1
            masks.append(pos.flatten())
            vis_labels.append(vises_one_hot)

        return images, pos_labels, masks, vis_labels

def sub_inference(x, output_num, train_mode):
    conv1_1 = conv_layer(x, 3, 64, 3, 1, 'conv1_1', train_mode=train_mode)
    conv1_2 = conv_layer(conv1_1, 64, 64, 3, 1, 'conv1_2', train_mode=train_mode)
    pool1 = max_pool(conv1_2, 2, 2, 'pool1')

    conv2_1 = conv_layer(pool1, 64, 128, 3, 1, 'conv2_1', train_mode=train_mode)
    conv2_2 = conv_layer(conv2_1, 128, 128, 3, 1, 'conv2_2', train_mode=train_mode)
    pool2 = max_pool(conv2_2, 2, 2, 'pool2')

    conv3_1 = conv_layer(pool2, 128, 256, 3, 1, 'conv3_1', train_mode=train_mode)
    conv3_2 = conv_layer(conv3_1, 256, 256, 3, 1, 'conv3_2', train_mode=train_mode)
    conv3_3 = conv_layer(conv3_2, 256, 256, 3, 1, 'conv3_3', train_mode=train_mode)
    pool3 = max_pool(conv3_3, 2, 2, 'pool3')

    conv4_1 = conv_layer(pool3, 256, 512, 3, 1, 'conv4_1', train_mode=train_mode)
    conv4_2 = conv_layer(conv4_1, 512, 512, 3, 1, 'conv4_2', train_mode=train_mode)
    conv4_3 = conv_layer(conv4_2, 512, 512, 3, 1, 'conv4_3', train_mode=train_mode)
    pool4 = max_pool(conv4_3, 2, 2, 'pool4')

    conv5_1 = conv_layer(pool4, 512, 512, 3, 1, 'conv5_1', train_mode=train_mode)
    conv5_2 = conv_layer(conv5_1, 512, 512, 3, 1, 'conv5_2', train_mode=train_mode)
    conv5_3 = conv_layer(conv5_2, 512, 512, 3, 1, 'conv5_3', train_mode=train_mode)
    pool5 = max_pool(conv5_3, 2, 2, 'pool5')
    pool5 = flatten(pool5)
    #　这样写无法收敛，只能降到３００左右，不知道什么鬼，函待解决
    # fc6 = fc_layer(pool5, 25088, 4096, 'fc6', train_mode=train_mode) # 25088 = ((224 / (2**5))**2) * 512
    # if train_mode is not None:
    #     fc6 = tf.cond(train_mode, lambda: tf.nn.dropout(fc6, KEEP_PROB), lambda: fc6)
    #
    # fc7 = fc_layer(fc6, 4096, 4096, 'fc7', train_mode=train_mode)
    # if train_mode is not None:
    #     fc7 = tf.cond(train_mode, lambda: tf.nn.dropout(fc7, KEEP_PROB), lambda: fc7)
    #
    # fc8_new = fc_layer(fc7, 4096, 14, 'fc8_new', bn=False, relu=False, train_mode=train_mode)

    fc6 = fc_layer(pool5, 25088, 4096, 'fc6', train_mode=train_mode)  # 25088 = ((224 / (2**5))**2) * 512
    fc7 = fc_layer(fc6, 4096, 4096, 'fc7', train_mode=train_mode)
    fc8_pos = fc_layer(fc7, 4096, output_num * 2, 'fc8_pos', bn=False, relu=False, dropout=False, train_mode=train_mode)

    #此处新加可见性输出层 维度为[output_num, batch_size, 3]
    visiblity = []
    for i in range(output_num):
        visiblity.append(fc_layer(fc7, 4096, 3, "fc_visiblity_" + str(i), bn=False, relu=False, dropout=False,
                                  train_mode=train_mode))

    return fc8_pos, tf.convert_to_tensor(visiblity)

def inference(x, output_num, train_mode):
    '''
    forward propagation
    :param x: data
    :param output_num: category
    :param train_mode: train_mode
    :return: tuple(pos,visiblity)
    '''
    with tf.variable_scope('stage_1'):
        prediction = sub_inference(x, output_num, train_mode)
    return prediction


def loss_func(prediction, output_num, pos_, vis_, mask):

    pos_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.multiply(prediction[0], mask), pos_)),axis=1) / 2, axis=0)

    vis_loss = 0
    for i in range(output_num):
        #此处标签维度问题，怀疑是tf操作不会使第二个维度消失
        vis_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction[1][i,:,:], labels=tf.argmax(vis_[:,i,:], 1)))

    loss = pos_loss + vis_loss

    return loss

def train(dataset):
    output_num = len(POINT_INDEX[dataset.category])

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    pos_ = tf.placeholder(tf.float32, shape=[None, output_num * 2])
    vis_ = tf.placeholder(tf.int8, shape=[None, output_num, 3])
    train_mode = tf.placeholder(tf.bool)
    mask = tf.placeholder(tf.float32, shape= [None, output_num * 2])

    # vgg = vgg16g.Vgg16('./test-save-10000.npy')
    # vgg.build(x, train_mode)
    # print(vgg.get_var_count())

    prediction = inference(x, output_num, train_mode)

    loss = loss_func(prediction, output_num, pos_, vis_, mask)

    with tf.name_scope("train_step"):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.98, staircase=True)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(CKPT_MODEL_PATH + dataset.category)
        if ckpt:
            pretrain_model_path = ckpt.model_checkpoint_path
        else:
            pretrain_model_path = './VGG_imagenet.npy'

        load_pretrain_model(pretrain_model_path, dataset.category, sess, saver, NUM_STAGE)

        for batch in range(BATCH_NUM + 1):
            minibatch_X, minibatch_pos, minibatch_mask, minibatch_vis = dataset.get_minibatch(dataset.get_minibatch_ids())

            _, temp_loss, step = sess.run([train_op, loss, global_step],
                                          feed_dict={x: minibatch_X, pos_: minibatch_pos, mask: minibatch_mask, vis_:minibatch_vis,
                                                    train_mode: True})

            if batch % 100 == 0:
                print("Cost after batch %i: %f" % (batch, temp_loss))

            if batch % 10000 == 0 and batch != 0:
                saver.save(sess, os.path.join(CKPT_MODEL_PATH + dataset.category, CKPT_MODEL_NAME), global_step=global_step)
                print ('Save model at Batch_{:d}'.format(batch))


def show_img(img,label):
    #label = np.squeeze(label, axis=0)
    label = np.array(label, np.int16)
    label[np.where(label[:] < 20)] = 0
    label[np.where(label[:] > 223)] = 223

    for i in range(int(label.shape[0] / 2)):

        x,y = int(label[i * 2]), int(label[i * 2 + 1])
        print('%dth key point:(%d, %d)'%(i+1, x, y))

        cv2.circle(img,(x,y), 2, (0,0,255),2)

    cv2.namedWindow('img')
    cv2.imshow('img', img)
    if cv2.waitKey(0) == 27:
        return


def test(dataset):
    output_num = len(POINT_INDEX[dataset.category])
    test_image_ids, _ = dataset.load_csv('test')

    imgs = []
    img_id = test_image_ids[160:180]
    for id in img_id:
        img = load_image(id,'test')
        assert (img.shape == (224, 224, 3))

        # image = np.reshape(img, (1, 224, 224, 3))
        imgs.append(img)
    print('test img number: ',len(imgs))

    with tf.name_scope('test_x'):
        x = tf.placeholder(tf.float32,[None, 224, 224, 3],name='x_input')
        train_mode = tf.placeholder(tf.bool)


    y = inference(x, output_num, train_mode)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CKPT_MODEL_PATH + dataset.category)
        if ckpt:
            print('load parameters from '+ ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

            pos = sess.run(y, feed_dict={x:imgs, train_mode:False})

            for i in range(len(img_id)):
                print(img_id[i])
                show_img(imgs[i],pos[i,:])



if __name__ == "__main__":

    dataset = Dataset(DATASET_PATH, CATEGORY[4])

    # images, labels, masks, visibles = dataset.get_minibatch(dataset.get_minibatch_ids())
    # for i in range(0,10):
    #     print('image shape: ', images[i].shape)
    #     print('image label: ', labels[i].shape)
    #     print('mask: ', masks[i])
    #     print('visible: ', visibles[i])
    #     show_img(images[i],labels[i])

    train(dataset)


    # test_image_ids,_ = dataset.load_csv('test')
    # img = load_test_img(test_image_ids[5])

    # cv2.namedWindow('1')
    # cv2.imshow('1',img)
    # cv2.waitKey(0)

    # test(dataset)
