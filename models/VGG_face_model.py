from base.base_model import BaseModel
import tensorflow as tf
from utils.utils import *
import numpy as np

class VGG_Face_Model(BaseModel):
    def __init__(self, config, net, input_size, num_classes):
        super(VGG_Face_Model, self).__init__(config)
        self.input_size = input_size
        self.num_classes = num_classes
        self.build_model(net)

    def build_model(self, net):
        # self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, [None,  *self.input_size])
        x = self.x

        #run through net containing mat structures
        i = 0
        self.layers = []
        for n in net:

            if (n.type == 'conv'):
                stride = n.stride
                weight = n.weights[0]
                bias = n.weights[1]
                if (n.name[:2] == 'fc'):
                    padding = 'VALID'
                    if (len(weight.shape) <= 2):
                        weight = weight[np.newaxis, np.newaxis, :,  :]
                else:
                    padding = 'SAME'

                x = tf.nn.conv2d(x, tf.constant(weight), strides=(1, stride, stride, 1), padding=padding)
                x = tf.nn.bias_add(x, bias)
            elif (n.type == 'pool'):
                stride = n.stride
                pool = n.pool
                x = tf.nn.max_pool(x, ksize=(1, pool[0], pool[1], 1), strides=(1, stride, stride, 1), padding='SAME')
            elif (n.type == 'relu'):
                x = tf.nn.relu(x)
            elif (n.type == 'softmax'):
                # softmax is pred_output
                self.out = tf.nn.softmax(tf.reshape(x, [-1, self.num_classes]))
            else:
                #ignore
                print("skipped layer: {}".format(n.name))
                pass
            self.layers.append(x)


    def init_saver(self):
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        pass
