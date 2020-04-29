import numpy as np
import cv2 as cv
import os  
import pdb
 


def avenue_generator(batch_size, data_dir):  

    images = []
    labels = []
    
    # read all images and samples and resize them at 128 x 128
    
    base_dir = data_dir 
    
    videos_names = os.listdir(base_dir)
    for video_name in videos_names:
        images_names = os.listdir(os.path.join(base_dir, video_name, 'images_5_5_0.80'))
        for image_name in images_names: 
            img = cv.imread(os.path.join(base_dir, video_name, 'images_5_5_0.80', image_name))
            img = cv.resize(img, (64, 64))
            images.append(img)
            # get the labels
            short_name = image_name[:-7]
            meta = np.loadtxt(os.path.join(base_dir, video_name, 'meta_5_5_0.80', short_name + ".txt"))
            labels.append(meta[-2])
    
    images = np.array(images, np.float32)
    labels = np.array(labels) 
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        avenue_generator(batch_size, data_dir),
        avenue_generator(batch_size, data_dir)
    )
