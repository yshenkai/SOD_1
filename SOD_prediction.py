#from keras.callbacks import ModelCheckpoint
from pre_model import *
from load_data import *
import time

train_folder="ECSSD"
image_folder="images"
label_folder="ground_truth_mask"
gen_dict=dict()
test_gen=generateor_train_image(gen_dict=gen_dict,batch_size=1,train_folder=train_folder,image_folder=image_folder,label_folder=label_folder,shuffle=True,seed=1)

my_model=get_model()

#model_checkpoint=ModelCheckpoint(filepath="MCFEM_member.h5",monitor='loss',verbose=1,save_best_only=True)
start_time=time.time()
evalude=my_model.evaluate_generator(test_gen,steps=1000)
end_time=time.time()-start_time
print(evalude)
print(end_time/5019)

