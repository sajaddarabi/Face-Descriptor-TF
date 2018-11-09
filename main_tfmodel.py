import tensorflow as tf

import numpy as np
from models.VGG_face_model import VGG_Face_Model
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from utils.utils import loadmat
from scipy.misc import imread, imresize

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    #load mat files
    if (args.mat == ""):
        print("missing vgg_mat file path.")
    mat_data = loadmat(args.mat)


    meta_var = mat_data['net']
    class_names = meta_var['classes']['description']
    num_classes = len(class_names)
    input_size = meta_var['normalization']['imageSize']
    img_average = meta_var['normalization']['averageImage']

    net = meta_var['layers'] # contains model

    # create tensorflow session
    sess = tf.Session()

    # create the model
    print("loading model....")
    model = VGG_Face_Model(config, net, input_size, num_classes)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # test model on image;
    img = imread('./data/ak.png', mode='RGB')
    img = imresize(img, input_size[:2])
    img = img - img_average
    sess.run(init)
    pred = sess.run([model.out], feed_dict={model.x: img[np.newaxis, :, :, :]})
    top1_class = np.argmax(pred[0])
    print("pred: {}".format(class_names[top1_class]))
if __name__ == '__main__':
    main()
