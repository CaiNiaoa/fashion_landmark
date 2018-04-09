# -*- coding: utf-8 -*-
# Only stage 1

# The Python standard libraries
import os

# The third-party libraries
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

import vgg16g
import network
# My libraries
# ...自己编写的模块在这里

DATASET_PATH = './dataset/'
NPY_MODEL_PATH = 'pretrain_model/VGG_imagenet.npy'
CKPT_MODEL_PATH = './model/'  # 模型保存的地址
CKPT_MODEL_NAME = 'key_points_location.ckpt'  # 模型保存的名字
LOG_PATH = './log/'  # 保存TensorBoard日志的地址，用来查看计算图和显示程序中可视化的参数

BATCH_SIZE = 16  # 一个batch的大小
NUM_BATCH = 80000  # batch数目
LEARNING_RATE = 0.001  # 学习率


columns_buf = [  # 结果文档的各列名称

    'image_id', 'image_category',
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right',
    'waistline_left',
    'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',
    'top_hem_right', 'waistband_left',
    'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out',
    'bottom_right_in', 'bottom_right_out'

]


# 1、搭建网络（输入->中间过程的处理算法以及损失函数构建->输出）
# 2、准备数据（读取数据->清理数据->将数据组织成网络的输入需要的形式；以及打包为batch形式喂给网络）

class Dataset(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_anno = self.load_csv(dataset_path, 'train')
        self.test_anno = self.load_csv(dataset_path, 'test')
        # 注意此处的切片需要加上copy()函数来执行深拷贝，否则train_img_ids和train_anno[:, 0]指向相同的内存，
        # 在对train_img_ids做shuffle时也会打乱标签train_anno中的image_id与labels的对应关系
        self.train_img_ids = self.train_anno[:, 0].copy()
        self.test_img_ids = self.test_anno[:, 0].copy()
        self.cur = 0
        self.var_dict = {}

    def load_csv(self, dataset_path, flag):
        csv_path = os.path.join(dataset_path, flag, 'Annotations', flag + '.csv')
        annotations = pd.read_csv(csv_path)
        annotations = np.array(annotations)  # 将pd.DataFrame数据结构转换为np.ndarray结构
        annotations = annotations[np.where(annotations[:, 1] == 'trousers')[0], :]  # tempory trousers only
        if flag == 'train':
            annotations = np.array([anno for anno in annotations if np.sum(anno[2:] != '-1_-1_-1') == 7])
        elif flag == 'test':
            pass

        return annotations

    def shuffle_train_img_ids(self):
        np.random.shuffle(self.train_img_ids)
        self.cur = 0

    def get_next_minibatch_inds(self):
        if self.cur + BATCH_SIZE >= len(self.train_img_ids):
            self.shuffle_train_img_ids()
        inds = np.arange(self.cur, self.cur + BATCH_SIZE)
        self.cur += BATCH_SIZE

        return inds

    def get_minibatch(self, img_ids):
        """
		函数功能：获取一个batch的数据用于训练
		"""
        # 1、从训练数据总数的范围内中随机获得的一个batch size的index
        # 2、通过获得的index从image ids中得到image的abspath（也是Annotation文件中的image_id）
        # 3、根据abspath加载image到内存中，同时从Annotation文件中加载对应image_id（同abspath）的label
        # 4、将image和label组成一个minibatch返回用于训练
        imgs = []
        labels = []
        for img_id in img_ids:
            # image
            img_path = os.path.join(DATASET_PATH, 'train', img_id)
            img = cv2.imread(img_path)
            assert (img.all() != None)
            h, w = img.shape[0], img.shape[1]  # The picture's format is HWC. x is W, y is H.
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            assert (img.shape == (224, 224, 3))
            h_k, w_k = 224.0 / h, 224.0 / w
            # label
            label = self.train_anno[np.where(self.train_anno[:, 0] == img_id)[0], 2:]
            label = label[np.where(label[:] != '-1_-1_-1')]
            # label = np.array([[int(xy.split('_')[0]), int(xy.split('_')[1])] for xy in label]).flatten() # 原图中的坐标labels
            label = np.array([[int(int(xy.split('_')[0]) * w_k), int(int(xy.split('_')[1]) * h_k)] for xy in label]).flatten() # 缩放之后的坐标labels
            # label = np.array([[int(xy.split('_')[0]) / float(w), int(xy.split('_')[1]) / float(h)] for xy in label]).flatten() # 归一化：直接除以对应坐标方向的长度
            # label = np.array([[(int(int(xy.split('_')[0]) * w_k) - 112) / 224.0, (int(int(xy.split('_')[1]) * h_k) - 112) / 224] for xy in label]).flatten() # 论文中的归一化方法。可以化简为下式
            # label = np.array([[float(xy.split('_')[0]) / w - 0.5, float(xy.split('_')[1]) / h - 0.5] for xy in label]).flatten()  # 归一化
            assert (len(label) == 14)

            imgs.append(img)
            labels.append(label)
        # print ('Get One Batch*******************************************************************')

        return (imgs, labels)

    def get_next_minibatch(self):
        inds = self.get_next_minibatch_inds()
        img_ids = [self.train_img_ids[i] for i in inds]
        return self.get_minibatch(img_ids)


def batch_norm(inputs, is_training, is_conv_out, decay=0.999):
    scale = tf.get_variable('scale', initializer=tf.ones([inputs.get_shape()[-1]]))
    beta = tf.get_variable('beta', initializer=tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.get_variable('pop_mean', initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.get_variable('pop_var', initializer=tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training == True:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def fc(name, pre_layer, w_shape_1, relu=True, dropout=True, bn=True, train_flag=True, regularizer=tf.contrib.layers.l2_regularizer(0.001)):
    """
	函数功能：构建一个全连接层
	参数：
		name: 全连接层的名字
		pre_layer: 上一层的输出，也是当前层的输入
		w_shape_1: 当前层的权重的形状的第一维的大小，第零维的大小可由当前层的输入得到
		relu: 是否relu
		dropout: 是否dropout
		bn: 是否batch normal
		train_flag: 训练还是测试
		regularizer: 采用的正则化方式
	返回：当前层的输出
	"""
    with tf.variable_scope(name) as scope:
        w_shape = [pre_layer.get_shape()[1], w_shape_1]
        b_shape = w_shape_1
        weights = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(0.0, stddev=0.1), trainable=True)
        # if regularizer != None: tf.add_to_collection("regularizer_loss", regularizer(weights))  # 对weights进行L2正则化
        biases = tf.get_variable("biases", [b_shape], initializer=tf.constant_initializer(0.1), trainable=True)
        layer = tf.matmul(pre_layer, weights) + biases
        # if bn == True: layer = tf.layers.batch_norm(layer, training=True) # 对输出使用BN算法归一化
        if bn == True: layer = batch_norm(layer, is_training=train_flag, is_conv_out=False)
        if relu == True: layer = tf.nn.relu(layer)
        if train_flag == True and dropout == True: layer = tf.nn.dropout(layer, 0.5)
    return layer


def conv(name, pre_layer, kernel, strides=[1, 1, 1, 1], padding="SAME", bn=True, train_flag=True):
    """
	函数功能：构建一个卷积层
	参数：
		name: 卷积层的名字
		pre_layer: 上一层的输出，也是当前层的输入
		kernel: 卷积核的尺寸
		strides: 卷积核做卷积时的步长
		padding: 采用的填充方式
		bn: 是否batch normal
		train_flag: 训练还是测试
	返回：当前层的输出
	"""
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", kernel, initializer=tf.truncated_normal_initializer(0.0, stddev=0.1), trainable=True)
        biases = tf.get_variable("biases", [kernel[-1]], initializer=tf.constant_initializer(0.0), trainable=True)
        conv = tf.nn.conv2d(pre_layer, weights, strides, padding)
        conv = tf.nn.bias_add(conv, biases)
        # if bn == True: conv = tf.layers.batch_norm(conv, training=True) # 对输出使用BN算法归一化
        if bn == True: conv = batch_norm(conv, is_training=train_flag, is_conv_out=True)
        conv = tf.nn.relu(conv)
    return conv


def max_pool(name, pre_layer, k_h, k_w, s_h, s_w, padding="SAME"):
    """
	函数功能：构建一个池化层
	参数：
		name: 池化层的名字
		pre_layer: 上一层的输出，也是当前层的输入
		k_h: 池化层中过滤器尺寸的长
		k_w: 池化层中过滤器尺寸的宽
		s_h: 池化层中过滤器在长这个维度上的步长
		s_w: 池化层中过滤器在宽这个维度上的步长
		padding: 采用的填充方式
	返回：当前层的输出
	"""
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(pre_layer, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)
    return pool


def flatten(pre_layer):
    """
	函数功能：将卷积和池化之后的数据转化为全连接层的输入数据（通过将矩阵扁平化为一个一维数组）
	参数：
		pre_layer: 卷积和池化之后的数据
	返回：扁平化之后的数据
	"""
    with tf.name_scope("pool2fc"):
        pre_layer_shape = pre_layer.get_shape().as_list()
        nodes = pre_layer_shape[1] * pre_layer_shape[2] * pre_layer_shape[3]
        flattened_vec = tf.reshape(pre_layer, [-1, nodes])
    return flattened_vec


def sub_model(input_data, stage='stage_1', pre_stage_output=None, train_flag=True):
    """
    函数功能：构建sub model
    参数：
    input_data: 网络的输入
    stage: 网络的阶段
    pre_stage_output: 前一阶段的输出
    返回：网络的输出
    """
    x = conv("conv1_1", input_data, [3, 3, 3, 64], train_flag=train_flag)
    x = conv("conv1_2", x, [3, 3, 64, 64], train_flag=train_flag)
    x = max_pool("conv1_pool", x, 2, 2, 2, 2)

    x = conv("conv2_1", x, [3, 3, 64, 128], train_flag=train_flag)
    x = conv("conv2_2", x, [3, 3, 128, 128], train_flag=train_flag)
    x = max_pool("conv2_pool", x, 2, 2, 2, 2)

    x = conv("conv3_1", x, [3, 3, 128, 256], train_flag=train_flag)
    x = conv("conv3_2", x, [3, 3, 256, 256], train_flag=train_flag)
    x = conv("conv3_3", x, [3, 3, 256, 256], train_flag=train_flag)
    x = max_pool("conv3_pool", x, 2, 2, 2, 2)

    x = conv("conv4_1", x, [3, 3, 256, 512], train_flag=train_flag)
    x = conv("conv4_2", x, [3, 3, 512, 512], train_flag=train_flag)
    x = conv("conv4_3", x, [3, 3, 512, 512], train_flag=train_flag)
    x = max_pool("conv4_pool", x, 2, 2, 2, 2)

    x = conv("conv5_1", x, [3, 3, 512, 512], train_flag=train_flag)
    x = conv("conv5_2", x, [3, 3, 512, 512], train_flag=train_flag)
    x = conv("conv5_3", x, [3, 3, 512, 512], train_flag=train_flag)
    x = max_pool("conv5_pool", x, 2, 2, 2, 2)

    x = flatten(x)
    x = fc("fc6", x, 4096, relu=True, dropout=True, bn=True, train_flag=train_flag)
    if stage == 'stage_1':
        x = fc("fc7", x, 4096, relu=True, dropout=True, bn=True, train_flag=train_flag)
        # 先使用trousers（裤子）数据来构建模型（trousers共7个关键点），测试模型的有效性
        positions = fc("fc8_new", x, 14, relu=False, dropout=False, bn=False, train_flag=train_flag)
        return positions  # [visibility_1, visibility_2, visibility_3, visibility_4, visibility_5, visibility_6, visibility_7, visibility_8]
        # positions = tf.nn.softmax(positions, name="softmax")
        # visibility_1 = fc("fc_visibility_1", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_2 = fc("fc_visibility_2", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_3 = fc("fc_visibility_3", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_4 = fc("fc_visibility_4", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_5 = fc("fc_visibility_5", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_6 = fc("fc_visibility_6", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_7 = fc("fc_visibility_7", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)
        # visibility_8 = fc("fc_visibility_8", x, 3, relu=False, dropout=False, bn=False, train_flag=train_flag)

    elif stage == 'stage_2' or stage == 'stage_3':
        x_tmp = fc("fc_2or3_1", pre_stage_output, 512, relu=False, dropout=False, bn=False, train_flag=train_flag)
        x = tf.concat(1, (x, x_tmp))
        x = fc("fc_2or3_2", x, 4096, relu=True, dropout=True, bn=True, train_flag=train_flag)
        offsets = fc("fc_offsets", x, 14, relu=False, dropout=False, bn=False, train_flag=train_flag)
        return offsets  # [visibility_1, visibility_2, visibility_3, visibility_4, visibility_5, visibility_6, visibility_7, visibility_8]
        # visibility_1 = fc("fc_visibility_1", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_2 = fc("fc_visibility_2", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_3 = fc("fc_visibility_3", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_4 = fc("fc_visibility_4", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_5 = fc("fc_visibility_5", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_6 = fc("fc_visibility_6", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_7 = fc("fc_visibility_7", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)
        # visibility_8 = fc("fc_visibility_8", x, 3, relu=False, dropout=False, bn=False, train_flag=True, train_flag=train_flag)


def model(input_data, num_stage, train_flag=True):
	"""
	"""
	# one stage
	if num_stage == 1:
		with tf.variable_scope('stage_1'):
			positions = sub_model(input_data, train_flag=train_flag)
		return positions
	# # two stage
	# elif num_stage == 2:
	# 	with tf.variable_scope('stage_1'):
	# 		positions = sub_model(input_data, train_flag=train_flag)
	# 	with tf.variable_scope('stage_2'):
	# 		offsets = sub_model(input_data, stage='stage_2', pre_stage_output=positions, train_flag=train_flag)
	# 	return (positions, offsets)
	# # stage 3
	# # ...


def load_pretrain_model(sess, saver, pretrain_model_path, num_stage):
    if pretrain_model_path.split('/')[1] == 'model':
        saver.restore(sess, pretrain_model_path)
        print ('Load parameters from ' + pretrain_model_path)
    else:
        vgg16_npy = np.load(pretrain_model_path, encoding='latin1').item()  # load the pretrained parameters
        for key in vgg16_npy:
            for stage_i in range(1, num_stage + 1):
                with tf.variable_scope('stage_'+ str(stage_i), reuse=True):
                    with tf.variable_scope(key, reuse=True):
                        for subkey in vgg16_npy[key]:
                            try:
                                var = tf.get_variable(subkey)
                                sess.run(tf.assign(var, vgg16_npy[key][subkey]))
                                print('assign pretrain model ' + subkey + ' to stage ' + str(stage_i) + ' ' + key)
                                # name = ['weights', 'biases']
                                # var = tf.get_variable(name[subkey])
                                # sess.run(tf.assign(var, vgg16_npy[key][subkey]))
                                # print('assign pretrain model ' + name[subkey] + ' to stage ' + str(stage_i) + ' ' + key)
                            except ValueError:
                                print('ignore stage ' + str(stage_i) + ' ' + key)


def loss_func(logits, labels, num_stage):
    """
    函数功能：Euclidean loss function
    参数：
    logits: 网络的输出（先使用trousers子数据集）（如果只有stage 1，网络的输出只有positions，如果包含stage 2或stage 3，网络的输出除了positions还有offsets）
    labels: 标签（同x）
    """
    if num_stage == 1:
        euclidean_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(logits, labels)), axis=1) / 2, axis=0)
    elif num_stage == 2:
        positions_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(logits[0], labels)), axis=1) / 2, axis=0)
        stage_2_labels = tf.subtract(logits[0], labels)
        offsets_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(logits[1], stage_2_labels)), axis=1) / 2, axis=0)
        euclidean_loss = positions_loss + offsets_loss
    return euclidean_loss

def train(dataset):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 14], name="y-input")
        train_flag = tf.placeholder(tf.bool, name='train_flag')

    num_stage = 1
    # pre = model(x, num_stage=num_stage, train_flag=train_flag)  # forward
    with tf.variable_scope('stage_1'):
          pre = network.inference(x,train_mode=train_flag)
    with tf.name_scope("loss"):
        euclidean_loss = loss_func(pre, y_, num_stage=num_stage)
        # regularizer_loss = tf.add_n(tf.get_collection("regularizer_loss"))  # 得到L2正则化损失
        loss = euclidean_loss # + regularizer_loss  # 总损失

    with tf.name_scope("train_step"):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.98, staircase=True)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    # 定义保存网络参数的Saver
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #  加载预训练的模型参数
        ckpt = tf.train.get_checkpoint_state(CKPT_MODEL_PATH)
        if ckpt:
            pretrain_model_path = ckpt.model_checkpoint_path
        else:
            pretrain_model_path = os.path.join(DATASET_PATH, NPY_MODEL_PATH)
        load_pretrain_model(sess, saver, pretrain_model_path, num_stage=num_stage)

        for batch_i in range(NUM_BATCH + 1):
            batch_xs, batch_ys = dataset.get_next_minibatch()
            _, loss_val, step = sess.run([train_step, loss, global_step], {x: batch_xs, y_: batch_ys, train_flag: True})
            # print ('One Batch Done******************************************************************')

            if batch_i % 10000 == 0 and batch_i != 0:
                saver.save(sess, os.path.join(CKPT_MODEL_PATH, CKPT_MODEL_NAME), global_step=global_step)

                print ('Save model at Batch_{:d}'.format(batch_i))
            if batch_i % 100 == 0:
                print ('Batch_{:d}, Training loss: {:>3.4f}'.format(batch_i, loss_val))

def show_img(img,label):
    print(label.shape)
    for i in range(int(label.shape[0] / 2)):
        x,y = int(label[i * 2]), int(label[i * 2 + 1])

        if x > 223:
            x = 223
        if x < 0:
            x = 0
        if y > 223:
            y = 223
        if y < 0:
            y = 0
        print(x, ' ', y)
        cv2.circle(img,(x,y), 2, (0,0,255),2)

    cv2.namedWindow('img')
    cv2.imshow('img', img)
    if cv2.waitKey(0) == 27:
        return

def test(img_id):
    minibatch_X, minibatch_Y = dataset.get_next_minibatch()


    images = tf.placeholder(tf.float32, [None, 224, 224, 3])

    train_mode = tf.placeholder(tf.bool)
    num_stage = 1
    #pre = model(images, num_stage=num_stage, train_flag=train_mode)  # forward
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    #print(vgg.get_var_count())
    with tf.variable_scope('stage_1'):
          pre = network.inference(images,train_mode)
         #   pre = sub_model(images, train_flag=train_mode)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(CKPT_MODEL_PATH)
        if ckpt:
            pretrain_model_path = ckpt.model_checkpoint_path
        else:
            #pretrain_model_path = './VGG_imagenet.npy'
            pretrain_model_path = './test-save-10000.npy'

        load_pretrain_model(sess, saver, pretrain_model_path, num_stage=num_stage)

        prob = sess.run(pre, feed_dict={images: minibatch_X, train_mode: False})

        print(prob)
        for i in range(16):
            show_img(minibatch_X[i], prob[i,:])

        # show_img(img, prob[0,:])

def train1(dataset):
    # x = tf.placeholder("float", shape=[None, 224, 224, 3])
    # y_ = tf.placeholder("float", shape=[None, 14])
    # train_mode = tf.placeholder(tf.bool)
    #
    # vgg = vgg16g.Vgg16('./dataset/pretrain_model/vgg16.npy')
    # vgg.build(x, train_mode)
    # # print(vgg.get_var_count())
    #
    #
    # cost = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(vgg.fc8_new, y_)), axis=1) / 2, axis=0)
    # loss = loss_func(vgg.fc8_new, y_, 1)
    # train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 14], name="y-input")
        train_flag = tf.placeholder(tf.bool, name='train_flag')

    num_stage = 1
    #fc8_new = model(x, num_stage=num_stage, train_flag=train_flag)  # forward

    vgg = vgg16g.Vgg16('./dataset/pretrain_model/VGG_imagenet.npy')
    fc8_new = vgg.build(x, train_flag = train_flag)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.name_scope("loss"):
        euclidean_loss = loss_func(fc8_new, y_, num_stage=num_stage)
        # regularizer_loss = tf.add_n(tf.get_collection("regularizer_loss"))  # 得到L2正则化损失
        loss = euclidean_loss # + regularizer_loss  # 总损失

    with tf.name_scope("train_step"):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.98, staircase=True)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    # 定义保存网络参数的Saver
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # pretrain_model_path = os.path.join(DATASET_PATH, NPY_MODEL_PATH)
        # load_pretrain_model(sess, saver, pretrain_model_path, num_stage=num_stage)
        vgg.load_pretrain_model(sess,1)
        for batch in range(3000):
            batch_xs, batch_ys = dataset.get_next_minibatch()
            _, loss_val, step = sess.run([train_step, loss, global_step], {x: batch_xs, y_: batch_ys, train_flag: True})

            if batch % 100 == 0:
                print("Cost after batch %i: %f" % (batch, loss_val))
            if batch % 1500 == 0 and batch != 0:
                vgg.save_npy(sess, './test-save-1500.npy')



if __name__ == "__main__":
    dataset = Dataset(DATASET_PATH)
    train(dataset)
    # test(1)










