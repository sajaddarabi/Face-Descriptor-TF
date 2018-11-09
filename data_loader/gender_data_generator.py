import tensorflow as tf
import numpy as np
import os
import glob

class GenderDataGenerator:
    def __init__(self, config, sess, input_dim):
        self.image_height = input_dim[0]
        self.image_width = input_dim[1]
        self.config = config
        self.sess = sess
        self.fold_data_name = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt',
                               'fold_3_data.txt', 'fold_4_data.txt']
        self.data = []
        self.data_path = config.gender_data_path
        for fd in self.fold_data_name:
            self.data += self.load_data(os.path.join(self.data_path, fd))

        self.data = np.asarray(self.data)

        train_test_ratio = 0.8
        num_examples = len(self.data)
        self.train = self.data[:int(num_examples * train_test_ratio)]
        self.test = self.data[int(num_examples * train_test_ratio) + 1:]


        self.train_input = self.train[:, 0]
        self.train_y = self.train[:, 1]

        self.test_input = self.test[:, 0]
        self.test_y = self.test[:, 1]

    def load_data(self, path):
        gender = ['f', 'm']
        data = []
        with open(path, 'r') as f:
            #skip first line
            first = True
            for l in f:
                if (first):
                    first = False
                    continue
                split_line = l.split('\t')
                t = []

                t.append(split_line[0])
                t.append(split_line[1])
                t.append(split_line[4])
                img_path = os.path.join(self.data_path, t[0])
                files = glob.glob(img_path + "/*.jpg")
                for ff in files:
                    if t[1] in ff:
                        if (t[2] in gender):
                            data.append([ff, int(t[2] == 'f')])
        return data

    def get_test(self, batch_size):
        return self.test_input, self.test_y

    def proc_image(self, img):
        data = tf.placeholder(tf.string)
        jpg = tf.image.decode_jpeg(data, channels=3)
        resize = tf.image.resize_images(jpg, [self.image_height, self.image_width])
        with tf.gfile.FastGFile(img, 'rb') as f:
            img_data = f.read()
            img_data = self.sess.run(resize, feed_dict={data: img_data})
        return img_data
    def next_batch(self, batch_size):

        idx = np.random.choice(len(self.train_input), batch_size)
        x = self.train_input[idx]
        y = self.train_y[idx].astype(int)
        y = np.eye(2)[y]
        import pdb; pdb.set_trace()
        batch_x = []
        batch_y = []
        for i in range(len(x)):
            batch_x.append(self.proc_image(x[i]))
            batch_y.append(y[i])

        yield batch_x, batch_y
