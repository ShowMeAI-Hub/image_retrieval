import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate
from keras.layers.merge import add
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras import backend as K

from custom_layers import subtract, norm
from vgg16 import cifar10vgg
from utils import triplet_loss, accuracy, euclidean_distance


class TripletNet:
    '''
    Triplet网络模型
    '''

    def __init__(self, shape=(32, 32, 3), dimensions=128, pretrained=False,
                 learning_rate=0.001, momentum=0.9):
        if pretrained:
            self.model, self.orig, self.residual = self.build_pretrained_model(shape, dimensions)
        else:
            self.model, self.orig, self.residual = self.build_triplet_model(shape, dimensions)
            
        v_optimizer = SGD(lr=learning_rate, momentum=momentum, nesterov=True)
    
        self.model.compile(
                #loss='mean_squared_error',
                loss=triplet_loss,
                optimizer=v_optimizer,
                metrics=['accuracy'])
        
        self.fit = self.model.fit
        self.fit_generator = self.model.fit_generator
        self.predict = self.model.predict
        self.evaluate = self.model.evaluate
        self.summary = self.model.summary

    def build_triplet_model(self, shape, dimensions):
        
        net = self.build_embedding(shape, dimensions)

        # Receive 3 inputs
        # Decide which of the two alternatives is closest to the original
        # x - Original Image
        # x1 - Alternative 1
        # x2 - Alternative 2
        x  = Input(shape=shape, name='x')
        x1 = Input(shape=shape, name='x1')
        x2 = Input(shape=shape, name='x2')

        # 前向计算得到embedding表示
        net_anchor   = net(x)
        net_positive = net(x1)
        net_negative = net(x2)
        
        # 欧氏距离计算
        positive_dist = subtract(net_anchor, net_positive)
        negative_dist = subtract(net_anchor, net_negative)
        
        positive_dist = norm(positive_dist)
        negative_dist = norm(negative_dist)

        # 对比与计算
        out = Lambda(lambda a: a[0] - a[1])([positive_dist, negative_dist])
        return Model(inputs=[x, x1, x2], outputs=out), x, net_anchor
    
    
    def build_pretrained_model(self, shape, dimensions):
        
        # Using vgg16 pre-trained weights
        net = cifar10vgg(x_shape=shape, train=False).model
        for _ in range(7): net.pop()
        net.add(GlobalMaxPooling2D(name='gap'))
        net.add(Dense(dimensions, activation='relu'))
        #net.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
            
        #net = self.build_embedding(shape, dimensions)

        # Receive 3 inputs
        # Decide which of the two alternatives is closest to the original
        x =  Input(shape=shape, name='x')
        x1 = Input(shape=shape, name='x1')
        x2 = Input(shape=shape, name='x2')

        # Get the embedded values
        net_anchor   = net(x)
        net_positive = net(x1)
        net_negative = net(x2)
        
        # 计算欧氏距离
        positive_dist = subtract(net_anchor, net_positive)
        negative_dist = subtract(net_anchor, net_negative)

        positive_dist = norm(positive_dist)
        negative_dist = norm(negative_dist)

        # Compare
        out = Lambda(lambda a: a[0] - a[1])([positive_dist, negative_dist])
        return Model(inputs=[x, x1, x2], outputs=out), x, net_anchor

    def build_embedding(self, shape, dimensions):
        inp = Input(shape=shape)
        x = inp

        # 3 Conv + MaxPool + Relu w/ Dropout
        x = self.convolutional_layer(64, kernel_size=3, is_pool=False)(x)
        x = self.convolutional_layer(128, kernel_size=3)(x)
        x = self.convolutional_layer(128, kernel_size=3,is_pool=False)(x)
        x = self.convolutional_layer(256, kernel_size=3)(x)
        x = self.convolutional_layer(256, kernel_size=3,is_pool=False)(x)
        x = self.convolutional_layer(512, kernel_size=3)(x)

        # 1 最终的卷积层到128维embedding
        x = Conv2D(dimensions, kernel_size=2, padding='same')(x)
        x = GlobalMaxPooling2D(name='gmp')(x)
        #x = Dense(dimensions, activation='relu')(x)

        out = x
        return Model(inputs=inp, outputs=out)

    def convolutional_layer(self, filters, kernel_size, is_pool=True):
        def _layer(x):
            x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
            if is_pool:
                x = MaxPooling2D(pool_size=2)(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Dropout(0.25)(x)
            return x

        return _layer
    
      

if __name__ == "__main__":
    t = TripletNet(shape=(32, 32, 3), dimensions=128)
    t.model.summary()


