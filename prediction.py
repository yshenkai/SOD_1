from pre_model import *
from skimage import io,transform
import numpy as np
import glob
img_paths=glob.glob("DUT-OMRON/Img/*.jpg")
model = get_model()
for path in img_paths:
    img=io.imread(path)
    re_img=transform.resize(img,(256,256,3))
    re_img=np.reshape(re_img,(1,256,256,3))
    r_img=model.predict(re_img)
    r_img=np.reshape(r_img,(256,256))
    r_img=transform.resize(r_img,(img.shape[0],img.shape[1]))
    r_img=r_img*255
    c=np.array(r_img,np.uint8)
    img_save_path=path.split("/")[-1]
    io.imsave("DUT-OMRON_PRE/"+img_save_path,c)

