from base.base_model import BaseModel
import tensorflow as tf
from utils.utils import *
import numpy as np

from .VGG_face_model import VGG_Face_Model
from base.base_model import BaseModel
import tensorflow as tf


class VGG_Face_Transfer(BaseModel):
    def __init__(self, config):
        super(VGG_Face_Transfer, self).__init__(config)
        mat_data = loadmat(config.path_mat_file)
        meta_var = mat_data['net']
        self.input_size = meta_var['normalization']['imageSize']
        net = meta_var['layers']
        self.VGG_face_model = VGG_Face_Model(config, net, self.input_size, len(meta_var['classes']['description']))
        self.num_classes = 2
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.y = tf.placeholder(tf.float32, [None, self.num_classes])

        descriptors = self.VGG_face_model.layers[-3]
        logits = tf.squeeze(tf.layers.dense(descriptors, units=self.num_classes, name="dense134"))

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))


            correct_prediction = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32),
                                          tf.argmax(self.y, axis=1, output_type=tf.int32))

            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.variable_scope("train-metric"):
            ema = tf.train.ExponentialMovingAverage(0.997, self.global_step_tensor, name='average')
            self.ema_op = ema.apply([self.cross_entropy, self.accuracy] + tf.trainable_variables())
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema_op)
            self.loss_avg = ema.average(self.cross_entropy)

            self.accuracy_avg = ema.average(self.accuracy)

            check_loss = tf.check_numerics(self.cross_entropy, 'model diverged: loss->nan')
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)

        with tf.name_scope("train_step"):
            gradient_multipliers = None
            self.optimizer = tf.contrib.layers.optimize_loss(self.cross_entropy, self.global_step_tensor,
                                                             self.config.learning_rate,
                                                             self.config.optimizer,
                                                             gradient_noise_scale=None,
                                                             gradient_multipliers=gradient_multipliers,
                                                             clip_gradients=None,  # moving_average_decay=0.9,
                                                             learning_rate_decay_fn=self.learning_rate_decay_fn,
                                                             update_ops=None,
                                                             variables=None,
                                                             name=None)

        updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies([self.optimizer]):
            self.train_step = tf.group(*updates_collection)

    def learning_rate_decay_fn(self, learning_rate, global_step):
        """
        learning rate for optimizer
        """
        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                        global_step, 50000, 0.5, staircase=True)
        return learning_rate


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        pass
