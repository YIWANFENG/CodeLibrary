# -*- coding: utf-8 -*-
#本文件描述了一个AlexNet网络结构(5层卷积，3层全链接)，以供示范或者修改使用。
# pylint: disable=unused-import


import os
import AlexNet_helper #by me
import tensorflow as tf

images_train,labels_train = load_data("train")
#The shape of data is (224,224,3) 
print("Image total num:%g ,Labels total num: %g" %(len(images_train),len(labels_train)))



#权重初始化函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial)


images_input = tf.placeholder("float",[None,224,224,3])
labels_input = tf.placeholder("float",[None,1000])  #43种标签
learn_rate = tf.placeholder("float") 
keep_prob_1 = tf.placeholder("float")
keep_prob_2 = tf.placeholder("float")

#################### 权重(卷积核)定义 #################
#conv_1 
w_conv_1 = weight_variable([11,11,3,96])
b_conv_1 = bias_variable([96])

#pooling_1
ksize_pool_1 = [1,3,3,1]
strides_pool_1 = [1,2,2,1]

#conv_2
w_conv_2 = weight_variable([5,5,96,256])
b_conv_2 = bias_variable([256])

#pooling_2
ksize_pool_2 = [1,3,3,1]
strides_pool_2 = [1,2,2,1]

#conv_3 
w_conv_3 = weight_variable([3,3,256,384])
b_conv_3 = bias_variable([384])

#conv_4
w_conv_4 = weight_variable([3,3,384,384])
b_conv_4 = bias_variable([384])

#conv_5
w_conv_5 = weight_variable([3,3,384,256])
b_conv_5 = bias_variable([256])

#pooling_3 
ksize_pool_3 = [1,3,3,1]
strides_pool_3 = [1,2,2,1]


#full_connected_1
w_full_1 = weight_variable([6*6*256,4096])
b_full_1 = bias_variable([4096])

#full_connected_2
w_full_2 = weight_variable([4096,4096])
b_full_2 = bias_variable([4096])

# full_connected_2 (output)
w_out_3 = weight_variable([4096,1000])
b_out_3 = bias_variable([1000])

################### 模型层次定义 #################
### 1 conv  Input(None,224,224,3) Output(None,55,55,96)  ceil[(224-11+1)/4]=55
con1 = tf.nn.conv2d(images_input, w_conv_1, strides=[1,4,4,1], padding="VALID")
layer1 = tf.nn.relu(con1 + b_conv_1)
layer1 = tf.nn.local_response_normlization(layer1,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
# Max Poolinh Output(None,27,27,96)
layer1 = tf.nn.max_pool(layer1, ksize_pool_1, strides_pool_1, padding="SAME")

#### 2 conv Output(None,27,27,256)
con2 = tf.nn.conv2d(layer1, w_conv_2, strides=[1,1,1,1], padding="SAME")
layer2 = tf.nn.relu(con2 + b_conv_2)
layer2 = tf.nn.local_response_normlization(layer2,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
#  Max Pooling Output(None,13,13,256)
layer2 = tf.nn.max_pool(layer2, ksize_pool_2, strides_pool_2, padding="VALID")

#### 3 conv Output(None,13,13,384)
con3 = tf.nn.conv2d(layer2, w_conv_3, strides=[1,1,1,1], padding="VALID")
layer3 = tf.nn.relu(con3 + b_conv_3)

#### 4 conv Output(None,13,13,384)
con4 = tf.nn.conv2d(layer3, w_conv_4, strides=[1,1,1,1], padding="SAME")
layer4 = tf.nn.relu(con4 + b_conv_4)

### 5 conv Output(None,13,13,256)
con5 = tf.nn.conv2d(layer4, w_conv_5, strides=[1,1,1,1], padding="SAME")
layer5 = tf.nn.relu(con5 + b_conv_5)

#  Max Pooling Output(None,6,6,256)
layer5 = tf.nn.max_pool(layer5, ksize_pool_3, strides_pool_3, padding="VALID")

#### 6 full connected Input(None,9216) Output(None,4096)
layer5 = tf.reshape(layer5,[-1,9216])
full1 = tf.matmul(layer5, w_full_1)
layer6 = tf.nn.relu(full1 + b_full_1)
#Droput
layer6 = tf.nn.dropout(layer6, keep_prob_1)

#### 7 full connected Input(None,4096) Output(None,4096)
full2 = tf.matmul(layer6, w_full_2)
layer7 = tf.nn.relu(full2 + b_full_2)
#Droput
layer7 = tf.nn.dropout(layer7, keep_prob_2)

#### 8 Output  Output(None,43)
output = tf.matmul(layer7, w_out_3)
predicted_labels = tf.nn.softmax(output + b_out_3)

#loss
cross_entropy = -tf.reduce_sum( labels_input*tf.log(predicted_labels) )

#优化器
my_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

#正确率计算
correct_predict = tf.equal(tf.argmax(predicted_labels,1),tf.argmax(labels_input))
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float"))

#print("cross_entropy ",cross_entropy)
#print("predicted_labels ", predicted_labels)
############ 训练 #######################
train_set = DataGenerator(images_train,labels_train)

train_steps = 10000
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
print("Train start")
for i in range(train_steps):
  #获得一个训练块
  batch_images,batch_labels = train_set.get_next_batch(500)
  
  feed_dcit = { images_input:batch_images, labels_input:batch_labels ,learn_rate = 0.001 }
  _,loss,acc = sess.run([my_optimizer,cross_entropy,accuracy], 
                       feed_dict = feed_dict)
  if i% 10 == 0:
    print("step %d, training accuracy %g" %(i, acc))

print("Train end")


'''
#This part is just for test.
images_test,labels_test = load_data("test")
test_set = DataGenerator(images_test,labels_test)

print("Test start")

batch_images,batch_labes = test_set.get_next_batch(len(labels_test))
feed_dcit_test = { images_input:batch_images, labels_input:batch_labels }

acc_t,loss_t = sess.run([accuracy,cross_entropy],feed_dict = feed_dcit_test)

print("test cross_entropy %g,test accuracy %g" %(loss_t,acc_t))

print("Test end")
'''
sess.close()
