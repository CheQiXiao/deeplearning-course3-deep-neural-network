#!/user/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases #参见资料包，或者在文章底部copy
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import lr_utils #参见资料包，或者在文章底部copy

#1、两层  初始化参数
def initializa_parameters(n_x,n_h,n_y):
    """
    为了初始化两层网络参数而使用的函数

    :param n_x: 输入层节点
    :param n_h: 隐藏层节点
    :param n_y: 输出层节点
    :return: w b的参数字典
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）
    """
    W1=np.random.rand(n_h,n_x)*0.01
    W2=np.random.rand(n_y,n_h)*0.01
    b1=np.zeros((n_h,1))
    b2=np.zeros((n_y,1))

    assert (W1.shape == (n_h,n_x))
    assert (W2.shape == (n_y,n_h))
    assert (b1.shape == (n_h,1))
    assert (b2.shape == (n_y,1))
    parameters={
        "W1":W1,
        "W2":W2,
        "b1":b1,
        "b2":b2
    }
    return parameters
# print("==============测试initialize_parameters==============")
# parameters=initializa_parameters(3,2,1)
# print("W1="+str(parameters["W1"]))
# print("b1="+str(parameters["b1"]))
# print("W2="+str(parameters["W2"]))
# print("b2="+str(parameters["b2"]))
#1、多层
def initialize_parameters_deep(layers_dims):
    """
    用于求多层神经网络的参数
    :param layers_dims: 每个层的节点数的列表 如：layers_dims[1]即为第一层的节点数
    :return:
    """
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.rand(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1]) #??????为什么要除以前一层节点的开平方
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

        assert (parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert (parameters["b"+str(l)].shape==(layers_dims[l],1))

    return parameters
#测试initialize_parameters_deep
# print("==============测试initialize_parameters_deep==============")
# layers_dim=[5,4,3,2]
# parameters=initialize_parameters_deep(layers_dim)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("W3 = " + str(parameters["W3"]))
# print("b3 = " + str(parameters["b3"]))

#实现线性计算
def line_forward(A,W,b):
    """
    实现前向传播的线性部分

    :param A:上一层的激活，维度为（上一层的节点数量，示例的数量）
    :param W:
    :param b:
    :return:Z  激活功能的输入，也称为预激活参数
            linear_cache: 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
    """
    Z=np.dot(W,A)+b
    assert (Z.shape==(W.shape[0],A.shape[1]))# 矩阵W和矩阵A相乘为Z
    linear_cache=(A,W,b)

    return Z,linear_cache
#测试linear_forward
# print("==============测试linear_forward==============")
# A,W,b=testCases.linear_forward_test_case();
# Z,linear_cache=line_forward(A,W,b)
# print("A="+str(A))
# print("z="+str(Z))

#2、单层前向传播
def linear_activation_forward(A_prev,W,b,activation):
    """
         在上一个方法中实现了Z=W*X+b,在此方法中加上激活函数sigmoid or Relu
    :param A_prev: 上一层的激活，维度为(上一层的节点数量，示例数）
    :param W:
    :param b:
    :param activation:选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return:
        A - 激活函数的输出，也称为激活后的值
        cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
        linear_cache- 存着A,W，b,  也就是上一层的激活值A_prev以及W、b
        activation_cache 存着Z  Z=W*A_prev+b  activation_cache是linear_cache线性运算之后的值

    """
    if activation == "sigmoid":
        Z,linear_cache=line_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = line_forward(A_prev, W, b)
        A,activation_cache=relu(Z)
    assert (A.shape==(W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    return A,cache
#测试linear_activation_forward
# print("==============测试linear_activation_forward==============")
# A_prev,W,b=testCases.linear_activation_forward_test_case()
# A,cache=linear_activation_forward(A_prev,W,b,activation="sigmoid")
# print("sigmoid中="+str(A))
# A,cache=linear_activation_forward(A_prev,W,b,activation="relu")
# print("relu中="+str(A))

#2、实现多层激活（整个网络的激活）
def L_model_forward(X,parameters):
    """
    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
    也就是前L-1层是relu激活，最后一层用sigmoid激活
    :param X:
    :param parameters:initialize_parameters_deep（）的输出，由于多层，参数W和b也不确定有几个
    :return:
            AL： 表示sigmoid后最后的激活值，相当于最后的输出
            caches:
                 linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                 linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
    """
    caches=[]
    A=X
    L = len(parameters)//2  # 6 // 4 = 1  6 / 4 =1.5
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")  #已跳出循环，上一个激活值赋值给了A,而不是A_prev，所以此处用A
    caches.append(cache)

    assert (AL.shape == (1,X.shape[1]))
    return AL,caches
    # 测试L_model_forward
#print("==============测试L_model_forward==============")
# X,parameters=testCases.L_model_forward_test_case();
# AL,caches=L_model_forward(X,parameters)
# print("AL为"+str(AL))
# print("caches的长度为"+str(len(caches)))

#3、计算损失值
def compute_cost(AL,Y):
    """

    :param AL:
    :param Y:
    :return:
    """
    m=Y.shape[1]
    cost=-(1/m)*(np.sum(np.multiply(np.log(AL+1e-5),Y)+np.multiply(np.log(1-AL+1e-5),1-Y)))

    cost=np.squeeze(cost)
    assert (cost.shape ==())
    return cost
#测试compute_cost
# print("==============测试compute_cost==============")
# AL,Y=testCases.compute_cost_test_case()
# cost=compute_cost(AL,Y)
# print("cost = " + str(cost))
#
# 与前向传播类似，我们有需要使用三个步骤来构建反向传播：
#
# LINEAR 后向计算
# LINEAR -> ACTIVATION 后向计算，其中ACTIVATION 计算Relu或者Sigmoid 的结果
# [LINEAR -> RELU] ×\times× (L-1) -> LINEAR -> SIGMOID 后向计算 (整个模型)
#单层实现反向传播的线性部分
def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分（第L层）
    :param dZ:相对于（当前第l层的）线性输出的成本梯度
    :param cache:来自当前层前向传播的值的元组（A_prev，W，b）
    :return:
         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度，与W的维度相同
         db - 相对于b（当前层l）的成本梯度，与b维度相同
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev,dW,db
#测试linear_backward
# print("==============测试linear_backward==============")
# dZ,cache=testCases.linear_backward_test_case()
# dA_prev,dW,db=linear_backward(dZ,cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
#4、单层实现反向传播，L层是sigmoid或者relu的反向
def linear_activation_backward(dA,cache,activation):
    """

    :param dA: 当前层l的激活后的梯度值
    :param cache:我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
    :param activation: 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return: dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
         db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """
    linear_cache,activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    return dA_prev,dW,db
#测试linear_activation_backward
# print("==============测试linear_activation_backward==============")
# AL,linear_activation_cache=testCases.linear_activation_backward_test_case()
# dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="sigmoid")
# print("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " +str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

#4、多层实现反向传播，L层是sigmoid的反向，1到L-1是relu的反向
def L_model_backward(AL,Y,caches):
    """
        对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播
    :param AL:概率向量，正向传播的输出（L_model_forward（））
    :param Y:标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
    :param caches:包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层,里面存储的是relu之前的Z
                 linear_activation_forward（"sigmoid"）的cache,存储的是sigmoid之前的Z
                 cache存的是每层的 Z A_prev,W,b
    :return:
    """
    grads={}
    L=len(caches)  #也就是神经网络的深度
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL=-(np.divide(Y,AL))+(np.divide(1-Y,1-AL))     #见笔记

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)]=linear_activation_backward(dAL,current_cache,"sigmoid")#caches[cache[1] cache[2] cache[3]],caches的下标为0,1,2，caches[2]=cache[3]

    for l in reversed(range(L-1)):  #应该从后往前求
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA"+str(l+1)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp

    return grads
#测试L_model_backward
# print("==============测试L_model_backward==============")
# AL,Y_assess,caches=testCases.L_model_backward_test_case()
# grads=L_model_backward(AL,Y_assess,caches)
# print("dW="+str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))

#5、更新参数
def update_parameters(parameters,grads,learning_rate):
    """

    :param parameters:包含你的参数的字典
    :param grads:包含梯度值的字典，是L_model_backward的输出
    :param learning_rate:
    :return:
        parameters - 包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    """
    L=len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
#测试update_parameters
# print("==============测试update_parameters==============")
# parameters,grads=testCases.update_parameters_test_case()
# parameters=update_parameters(parameters,grads,learning_rate=0.1)
# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))

#两层神经网络训练模型
def two_layer_model(X,Y,layer_dim,learning_rate=0.0075,num_itearation=3000,print_cost=False,isPlot=True):
    """
    实现一个两层的神经网络，【LINEAR->RELU】 -> 【LINEAR->SIGMOID】
    :param X:
    :param Y:
    :param layer_dim:层数的向量，维度为(n_y,n_h,n_y)
    :param learning_rate:
    :param num_itearation:迭代的次数
    :param print_cost:是否打印成本值，每100次打印一次
    :param isPlot:是否绘制出误差值的图谱
    :return:
    """
    np.random.seed(1)
    grads={}
    costs=[]
    (n_x,n_h,n_y)=layer_dim

    """
    初始化参数
    """
    parameters=initializa_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    for i in range (0,num_itearation):
        #前向传播
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        #计算损失值
        cost=compute_cost(A2,Y)

        #反向传播
        dA2=-(np.divide(Y,A2))+(np.divide(1-Y,1-A2))

        ##向后传播，输入：“dA2，cache2，cache1”。 输出：“dA1，dW2，db2;还有dA0（未使用），dW1，db1”。
        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        ##向后传播完成后的数据保存到grads
        grads["dW1"]=dW1
        grads["db1"]=db1
        grads["dW2"]=dW2
        grads["db2"]=db2

        # 更新参数
        parameters=update_parameters(parameters,grads,learning_rate)
        W1=parameters["W1"]
        W2=parameters["W2"]
        b1=parameters["b1"]
        b2=parameters["b2"]

        # 打印成本值，如果print_cost=False则忽略
        if i%100==0:
            # 记录成本
            costs.append(cost)
            # 是否打印成本值
            if print_cost:
                print("第",i,"次迭代，成本为：",np.squeeze(cost))

    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per tens)')
        plt.title("Learning rate ="+str(learning_rate))
        plt.show()

    return parameters


#加载数据集
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_x=train_x_flatten/255
train_y=train_set_y

test_x_=test_x_flatten/255
test_y=test_set_y

#训练数据集
n_x=12288
n_h=7
n_y=1
layers_dims=(n_x,n_h,n_y)
parameters=two_layer_model(train_x,train_set_y,layers_dims,num_itearation=2500,print_cost=True,isPlot=True)

#正式预测
def predict(X,Y,parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层，通过已经训练完的模型对实际数据进行测试
    只需要使用前向传播得出最终预测值，将最终预测值与标签值对比得出准确率
    :param X:
    :param Y:
    :param parameters:
    :return:p - 给定数据集X的预测
    """
    m=X.shape[1]
    n=len(parameters)//2  #神经网络的层数
    p=np.zeros((1,m))

    """
    probas是最终的预测值
    """
    probas,caches=L_model_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    print("准确度为："+str(float(np.sum(p==Y))/m))

    return p
predictions_train=predict(train_x,train_y,parameters)
predictions_test=predict(test_x_,test_y,parameters)
