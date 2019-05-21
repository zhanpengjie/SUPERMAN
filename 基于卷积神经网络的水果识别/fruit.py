##导包
import os
import skimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import color,data,transform
from sklearn.utils import shuffle
import keras
from keras.utils import np_utils
##改变当前路径
os.chdir('D://fruits//fruits-360')
##获取对应路径文件夹下的图片数据
def load_data(dir_path):
    images=[] ##新建一个空列表用于存放图片数集
    labels=[] ##新建一个空列表用于存放标签数集
    #no5_imgs=[] ##新建一个空列表用于存放每个文件夹下的第五张图片
    #labels_no5=[] ##新建一个空列表用于存放第五张图片对应的文件夹名
    lab=os.listdir(dir_path)
    n=0
    for l in lab:
        img=os.listdir(dir_path+l) ##img为对应路径下的文件夹
        for i in img:
            img_path=dir_path+l+'/'+i ##是的话获取图片路径
            labels.append(int(n)) ##将外循环的迭代数n存于labels中
            images.append(skimage.data.imread(img_path)) ##读取对应路径图像存放于datasets中
        n+=1
    return images,labels ##返回的images内的图片存放顺序与实际文件夹中存放的顺序不同

##最终数据集与标签
images,labels=load_data('./Training/')
print(len(images),len(labels))
##最终测试数据集与标签
images_test,labels_test=load_data('./Test/')
print(len(images_test),len(labels_test))
##由于数据集过大,先加载小部分进行训练，n代表几个类
def load_small_data(dir_path,m):
    images_m=[] ##新建一个空列表用于存放图片数集
    labels_m=[] ##新建一个空列表用于存放标签数集
    lab=os.listdir(dir_path)
    n=0
    for l in lab:
        if(n>=m):
            break
        img=os.listdir(dir_path+l) ##img为对应路径下的文件夹
        for i in img:
            img_path=dir_path+l+'/'+i ##是的话获取图片路径
            labels_m.append(int(n)) ##将图片的上层文件夹转换为int类型存于labels中
            images_m.append(skimage.data.imread(img_path)) ##读取对应路径图像存放于images_m中
        n+=1
    return images_m,labels_m ## m类标签以及数据

images_20,labels_20=load_small_data('./Training/',40) ##训练集
images_test_20,labels_test_20=load_small_data('./Test/',40) ##测试集

##预处理数据函数（数组化，乱序）
def prepare_data(images,labels,n_classes):
    ##images64=cut_image(images,64,64) ##裁剪图片大小为64*64
    train_x=np.array(images)
    train_y=np.array(labels)
    ##images_gray=color.rgb2gray(images_a) ##转灰度
    indx=np.arange(0,train_y.shape[0])
    indx=shuffle(indx)
    train_x=train_x[indx]
    train_y=train_y[indx]
    train_y=keras.utils.to_categorical(train_y,n_classes) ##one-hot独热编码
    return train_x,train_y
##训练集数据预处理
train_x,train_y=prepare_data(images_20,labels_20,40)
##测试数据集与标签的数组化和乱序
test_x,test_y=prepare_data(images_test_20,labels_test_20,40)
## 配置神经网络的参数
n_classes=40 ##数据的类别数
batch_size=16 ##训练块的大小
kernel_h=kernel_w=5 ##卷积核尺寸
dropout=0.8 ##dropout的概率
depth_in=3 ##图片的通道数
depth_out1=64 ##第一层卷积的卷积核个数
depth_out2=128 ##第二层卷积的卷积核个数
image_size=train_x.shape[1] ##图片尺寸
n_sample=train_x.shape[0] ##训练样本个数
t_sample=test_x.shape[0] ##测试样本个数
 
##feed给神经网络的图像数据类型与shape,四维，第一维训练的数据量，第二、三维图片尺寸，第四维图像通道数
x=tf.placeholder(tf.float32,[None,100,100,3]) 
y=tf.placeholder(tf.float32,[None,n_classes]) ##feed到神经网络的标签数据的类型和shape
keep_prob=tf.placeholder(tf.float32) ##dropout的placeholder(解决过拟合)
fla=int((image_size*image_size/16)*depth_out2) ##用于扁平化处理的参数经过两层卷积池化后的图像大小*第二层的卷积核个数
##定义各卷积层和全连接层的权重变量
Weights={"con1_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_in,depth_out1])),
        "con2_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_out1,depth_out2])),
        "fc_w1":tf.Variable(tf.random_normal([int((image_size*image_size/16)*depth_out2),1024])),
        "fc_w2":tf.Variable(tf.random_normal([1024,512])),
        "out":tf.Variable(tf.random_normal([512,n_classes]))}
 
##定义各卷积层和全连接层的偏置变量
bias={"conv1_b":tf.Variable(tf.random_normal([depth_out1])),
      "conv2_b":tf.Variable(tf.random_normal([depth_out2])),
      "fc_b1":tf.Variable(tf.random_normal([1024])),
      "fc_b2":tf.Variable(tf.random_normal([512])),
      "out":tf.Variable(tf.random_normal([n_classes]))}

## 定义卷积层的生成函数
def conv2d(x,W,b,stride=1):
    x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
 
## 定义池化层的生成函数
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")
 
## 定义卷积神经网络生成函数
def conv_net(x,weights,biases,dropout):
    
    ## Convolutional layer 1(卷积层1)
    conv1 = conv2d(x,Weights['con1_w'],bias['conv1_b']) ##100*100*64
    conv1 = maxpool2d(conv1,2) ##经过池化层1 shape：50*50*64
     
    ## Convolutional layer 2（卷积层2）
    conv2 = conv2d(conv1,Weights['con2_w'],bias['conv2_b']) ##50*50*128
    conv2 = maxpool2d(conv2,2) ##经过池化层2 shape:25*25*128
    ## Fully connected layer 1(全连接层1)
    flatten = tf.reshape(conv2,[-1,fla]) ##Flatten层，扁平化处理
    fc1 = tf.add(tf.matmul(flatten,Weights['fc_w1']),bias['fc_b1'])
    fc1 = tf.nn.relu(fc1) ##经过relu激活函数
    print(flatten.get_shape())
    ## Fully connected layer 2(全连接层2)
    fc2 = tf.add(tf.matmul(fc1,Weights['fc_w2']),bias['fc_b2']) ##计算公式：输出参数=输入参数*权值+偏置
    fc2 = tf.nn.relu(fc2) ##经过relu激活函数
    
    ## Dropout（Dropout层防止预测数据过拟合）
    fc2 = tf.nn.dropout(fc2,dropout)
    ## Output class prediction
    prediction = tf.add(tf.matmul(fc2,Weights['out']),bias['out']) ##输出预测参数
    return prediction

## 优化预测准确率
prediction=conv_net(x,Weights,bias,keep_prob) ##生成卷积神经网络
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y)) ##交叉熵损失函数
optimizer=tf.train.AdamOptimizer(0.0009).minimize(cross_entropy) ##选择优化器以及学习率
##optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
##optimizer=tf.train.AdagradOptimizer(0.001).minimize(cross_entropy) ##选择优化器以及学习率
 
## 评估模型
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##训练块数据生成器
def gen_small_data(inputs,batch_size):
    i=0
    while True:
        small_data=inputs[i:(batch_size+i)]
        i+=batch_size
        yield small_data

## 初始会话并开始训练过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()   
    for  i  in range(10):
        train_x,train_y=prepare_data(images_20,labels_20,40) ##重新预处理数据
        train_x=gen_small_data(train_x,batch_size) ##生成图像块数据
        train_y=gen_small_data(train_y,batch_size) ##生成标签块数据
        for j in range(int(n_sample/batch_size)+1):
            x_=next(train_x) 
            y_=next(train_y)
            ##准备验证数据
            validate_feed={x:x_,y:y_,keep_prob:0.8}
            if i % 1 == 0:
                sess.run(optimizer, feed_dict=validate_feed)
                loss,acc = sess.run([cross_entropy,accuracy],feed_dict={x:x_,y:y_,keep_prob:0.8})
                print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss),"Training accuracy","{:.5f}".format(acc))
    print('Optimization Completed')
    ##准备测试数据
    test_x=test_x[0:400]
    test_y=test_y[0:400]
    test_feed={x:test_x,y:test_y,keep_prob: 0.8} 
    y1 = sess.run(prediction,feed_dict=test_feed)
    test_classes = np.argmax(y1,1)
    print('Testing Accuracy:',sess.run(accuracy,feed_dict=test_feed))
