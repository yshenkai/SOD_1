from keras.preprocessing.image import ImageDataGenerator
#from skimage import io,color
from keras.layers import UpSampling2D
import numpy as np
def adjust_data(img,label):
    img=img/255
    label=label/255
    label[label<0.5]=0
    label[label>=0.5]=1
    '''
    label=label[:,:,:,0] if len(label.shape)==4 else label[:,:,0]
    new_label=np.zeros(shape=(label.shape+(2,)))
    new_label[label==0,0]=1
    new_label[label==1,1]=1
    label=new_label
    '''
    return img,label
def generateor_train_image(gen_dict,batch_size,train_folder,image_folder,label_folder,shuffle=False,seed=1):
    img_gen=ImageDataGenerator(**gen_dict)
    label_gen=ImageDataGenerator(**gen_dict)
    img_train=img_gen.flow_from_directory(directory=train_folder,target_size=(256,256),color_mode='rgb',classes=[image_folder],class_mode=None,batch_size=batch_size,shuffle=shuffle,seed=seed,save_to_dir=None)
    label_train=label_gen.flow_from_directory(directory=train_folder,target_size=(256,256),color_mode="grayscale",classes=[label_folder],class_mode=None,batch_size=batch_size,shuffle=shuffle,seed=seed,save_to_dir=None)
    train_gen=zip(img_train,label_train)
    for img,label in train_gen:
        img,label=adjust_data(img,label)
        yield img,label
