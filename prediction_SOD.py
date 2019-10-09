from model import *
from skimage import io,transform
import numpy as np
img=io.imread("ILSVRC2012_test_00000250.jpg")
re_img=transform.resize(img,(256,256,3))
re_img=np.reshape(re_img,(1,256,256,3))
#print(re_img)
model=get_model()

c=model.predict(re_img)
c=np.reshape(c,(256,256))
c=c*255
c=np.array(c,dtype=np.uint8)
io.imsave("ILSVRC2012_test_00000250_7_predict.jpg",c)
