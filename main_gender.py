import tensorflow as tf

import numpy as np
from models.VGG_face_transfer import VGG_Face_Transfer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from utils.utils import loadmat
from scipy.misc import imread, imresize
from data_loader.gender_data_generator import GenderDataGenerator
from trainers.gender_trainer import GenderTrainer
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
    if (args.mat == '' and "path_mat_file" not in config.keys()):
        print("missing vgg_mat file path.")
    mat_data = loadmat(config.path_mat_file)
    meta_var = mat_data['net']
    class_names = meta_var['classes']['description']
    num_classes = len(class_names)
    input_dim = meta_var['normalization']['imageSize']
    img_average = meta_var['normalization']['averageImage']

    net = meta_var['layers'] # contains model

    # create tensorflow session
    sess = tf.Session()

    # create the model
    print("loading model....")
    model = VGG_Face_Transfer(config)

    #load data
    data = GenderDataGenerator(config, sess, input_dim)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = GenderTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # train
    trainer.train()
if __name__ == '__main__':
    main()
