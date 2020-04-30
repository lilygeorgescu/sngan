import numpy as np
import cv2 as cv
import os  
import pdb
 


class avenue_generator:
    
    def __init__(self, batch_size, data_dir):  

        images = []
        labels = []
    
        # read all images and samples and resize them at 128 x 128
    
        base_dir = data_dir 
        self.batch_size = batch_size
    
        videos_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        # pdb.set_trace()
        for video_name in videos_names:
            images_names = os.listdir(os.path.join(base_dir, video_name, 'meta_5_5_0.80'))
            for image_name in images_names: 
                img = cv.imread(os.path.join(base_dir, video_name, 'images_5_5_0.80', image_name[:-4] + "_01.png"))
                img = cv.resize(img, (64, 64))
                images.append(img)
                # get the labels
                short_name = image_name[:-7]
                meta = np.loadtxt(os.path.join(base_dir, video_name, 'meta_5_5_0.80', image_name))
                labels.append(meta[-2])
        print('images has been read') 
        self.images = np.array(images, np.float32)
        self.labels = np.array(labels) 
        self.index = 0

    def next(self):
        # pdb.set_trace()        
        if self.index == len(self.images) // self.batch_size:
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
        i = self.index
        self.index += 1
        # for i in range(int(len(images) / batch_size)):
        return (self.images[i * self.batch_size:(i + 1) * self.batch_size], self.labels[i * self.batch_size:(i + 1) * self.batch_size])

    


def load(batch_size, data_dir):
    return (
        avenue_generator(batch_size, data_dir),
        avenue_generator(batch_size, data_dir)
    )
