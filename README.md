# Face-Descriptor-TF

To run the codes, first `pip install -r requirements.txt`. Preferably in a virtual environment.

Then, download the vgg_face_matconvnet.tar.gz file from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). Download the [face_gender dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender). Once the dataset is extracted, the faces folder should also contain the following 5 .txt files

    - fold_0_data.txt
    - fold_1_data.txt
    - fold_2_data.txt
    - fold_3_data.txt
    - fold_4_data.txt

Add the paths to the config/config.json directory,
    - "gender_data_path": <path to gender dataset downloaded earlier>
    - "path_mat_file": <path to vgg descriptor model stored in vgg_face_matconvnet.tar.gz>
Other parameters can be changed here accordingly.

To run the classifier model run the following command (make sure you're in the directory that the main files are stored):

`python main_gender.py -c ./configs/config.json`


The project contains the following structure:


- Model directory contains the tensorflow model for VGG descriptor, as well as a model that uses this as a feature extractor to do gender classification.
- Trainer contains the training logic
- data_loader contains data pipeline
- data folder to store data (not necessary)
- Utils are anything additional that doesn't fit in the other folders.
