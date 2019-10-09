from keras.callbacks import ModelCheckpoint
from model import *
from load_data import *

train_folder="DUTS-TR"
image_folder="DUTS-TR-Image"
label_folder="DUTS-TR-Mask"
gen_dict=dict()
train_gen=generateor_train_image(gen_dict=gen_dict,batch_size=4,train_folder=train_folder,image_folder=image_folder,label_folder=label_folder,shuffle=True,seed=1)

my_model=get_model()

model_checkpoint=ModelCheckpoint(filepath="weights.{epoch:02d}.h5",monitor='loss',verbose=1,save_best_only=True)
my_model.fit_generator(train_gen,steps_per_epoch=2638,epochs=50,callbacks=[model_checkpoint],initial_epoch=40)


