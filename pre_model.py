from keras.models import Model
from keras.layers import Input,Conv2D,AtrousConv2D,Activation,MaxPooling2D,Add,Multiply,UpSampling2D,concatenate,GlobalAveragePooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.initializers import glorot_normal
import keras.backend as K
from keras import metrics
from Atten import *
def get_model():
    inputs=Input(shape=(256,256,3))
    conv1_1=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(inputs)
    bn1_1=BatchNormalization()(conv1_1)
    ac1_1=Activation("relu")(bn1_1)
    conv1_2=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac1_1)
    bn1_2=BatchNormalization()(conv1_2)
    ac1_2=Activation("relu")(bn1_2)
    atten1=Atten()(ac1_2)

    pool1=MaxPooling2D(pool_size=(2,2))(atten1)
    conv2_1=Conv2D(filters=128,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(pool1)
    bn2_1=BatchNormalization()(conv2_1)
    ac2_1=Activation("relu")(bn2_1)
    conv2_2=Conv2D(filters=128,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac2_1)
    bn2_2=BatchNormalization()(conv2_2)
    ac2_2=Activation("relu")(bn2_2)
    atten2=Atten()(ac2_2)

    pool2=MaxPooling2D(pool_size=(2,2))(atten2)
    conv3_1=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(pool2)
    bn3_1=BatchNormalization()(conv3_1)
    ac3_1=Activation("relu")(bn3_1)
    conv3_2=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac3_1)
    bn3_2=BatchNormalization()(conv3_2)
    ac3_2=Activation("relu")(bn3_2)
    conv3_3=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac3_2)
    bn3_3=BatchNormalization()(conv3_3)
    ac3_3=Activation("relu")(bn3_3)
    atten3=Atten()(ac3_3)

    pool3=MaxPooling2D(pool_size=(2,2))(atten3)
    conv4_1=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(pool3)
    bn4_1=BatchNormalization()(conv4_1)
    ac4_1=Activation("relu")(bn4_1)
    conv4_2=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac4_1)
    bn4_2=BatchNormalization()(conv4_2)
    ac4_2=Activation("relu")(bn4_2)
    conv4_3=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac4_2)
    bn4_3=BatchNormalization()(conv4_3)
    ac4_3=Activation("relu")(bn4_3)
    atten4=Atten()(ac4_3)

    pool4=MaxPooling2D(pool_size=(2,2))(atten4)
    conv5_1=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(pool4)
    bn5_1=BatchNormalization()(conv5_1)
    ac5_1=Activation("relu")(bn5_1)
    conv5_2=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac5_1)
    bn5_2=BatchNormalization()(conv5_2)
    ac5_2=Activation("relu")(bn5_2)
    conv5_3=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac5_2)
    bn5_3=BatchNormalization()(conv5_3)
    ac5_3=Activation("relu")(bn5_3)
    atten5=Atten()(ac5_3)

    global_conv5=Conv2D(filters=512,kernel_size=1,padding="same",kernel_initializer=glorot_normal())(atten5)
    bn_global5=BatchNormalization()(global_conv5)
    ac_global5=Activation("relu")(bn_global5)
    uatten4=UpSampling2D(size=2)(ac_global5)
    uatten3=UpSampling2D(size=2)(uatten4)
    uatten2=UpSampling2D(size=2)(uatten3)
    uatten1=UpSampling2D(size=2)(uatten2)

    u1_0=Conv2D(filters=3,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(uatten1)
    u1_0_0=Add()([BatchNormalization(axis=3)(inputs),BatchNormalization(axis=3)(u1_0)])
    ac_u1_0=Activation("relu")(u1_0_0)
    u1_1=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u1_0)
    bn_u1_1=BatchNormalization()(u1_1)
    ac_u1_1=Activation("relu")(bn_u1_1)
    u1_2=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u1_1)
    bn_u1_2=BatchNormalization()(u1_2)
    ac_u1_2=Activation("relu")(bn_u1_2)
    t_atten1=Atten()(ac_u1_2)

    t_pool1=MaxPooling2D(pool_size=2)(t_atten1)
    u2_0=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(uatten2)
    u2_0_0=Add()([BatchNormalization(axis=3)(t_pool1),BatchNormalization(axis=3)(u2_0)])
    ac_u2_0=Activation("relu")(u2_0_0)
    u2_1=Conv2D(filters=128,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u2_0)
    bn_u2_1=BatchNormalization()(u2_1)
    ac_u2_1=Activation("relu")(bn_u2_1)
    u2_2=Conv2D(filters=128,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u2_1)
    bn_u2_2=BatchNormalization()(u2_2)
    ac_u2_2=Activation("relu")(bn_u2_2)
    t_atten2=Atten()(ac_u2_2)

    t_pool2=MaxPooling2D(pool_size=2)(t_atten2)
    u3_0=Conv2D(filters=128,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(uatten3)
    u3_0_0=Add()([BatchNormalization(axis=3)(t_pool2),BatchNormalization(axis=3)(u3_0)])
    ac_u3_0=Activation("relu")(u3_0_0)
    u3_1=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u3_0)
    bn_u3_1=BatchNormalization()(u3_1)
    ac_u3_1=Activation("relu")(bn_u3_1)
    u3_2=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u3_1)
    bn_u3_2=BatchNormalization()(u3_2)
    ac_u3_2=Activation("relu")(bn_u3_2)
    u3_3=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u3_2)
    bn_u3_3=BatchNormalization()(u3_3)
    ac_u3_3=Activation("relu")(bn_u3_3)
    t_atten3=Atten()(ac_u3_3)

    t_pool3=MaxPooling2D(pool_size=2)(t_atten3)
    u4_0=Conv2D(filters=256,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(uatten4)
    u4_0_0=Add()([BatchNormalization(axis=3)(t_pool3),BatchNormalization(axis=3)(u4_0)])
    ac_u4_0=Activation("relu")(u4_0_0)
    u4_1=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u4_0)
    bn_u4_1=BatchNormalization()(u4_1)
    ac_u4_1=Activation("relu")(bn_u4_1)
    u4_2=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u4_1)
    bn_u4_2=BatchNormalization()(u4_2)
    ac_u4_2=Activation("relu")(bn_u4_2)
    u4_3=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u4_2)
    bn_u4_3=BatchNormalization()(u4_3)
    ac_u4_3=Activation("relu")(bn_u4_3)
    t_atten4=Atten()(ac_u4_3)

    t_pool4=MaxPooling2D(pool_size=2)(t_atten4)
    u5_0=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(global_conv5)
    u5_0_0=Add()([BatchNormalization(axis=3)(t_pool4),BatchNormalization(axis=3)(u5_0)])
    ac_u5_0=Activation("relu")(u5_0_0)
    u5_1=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u5_0)
    bn_u5_1=BatchNormalization()(u5_1)
    ac_u5_1=Activation("relu")(bn_u5_1)
    u5_2=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u5_1)
    bn_u5_2=BatchNormalization()(u5_2)
    ac_u5_2=Activation("relu")(bn_u5_2)
    u5_3=Conv2D(filters=512,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_u5_2)
    bn_u5_3=BatchNormalization()(u5_3)
    ac_u5_3=Activation("relu")(bn_u5_3)
    t_atten5=Atten()(ac_u5_3)


    art_conv5_1=Conv2D(filters=32,kernel_size=3,padding="same",kernel_initializer=glorot_normal(),dilation_rate=1)(t_atten5)
    bn_art5_1=BatchNormalization()(art_conv5_1)
    ac_art5_1=Activation("relu")(bn_art5_1)
    art_conv5_2=Conv2D(filters=32,kernel_size=3,padding="same",kernel_initializer=glorot_normal(),dilation_rate=3)(t_atten5)
    bn_art5_2=BatchNormalization()(art_conv5_2)
    ac_art5_2=Activation("relu")(bn_art5_2)
    art_conv5_3 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=5)(t_atten5)
    bn_art5_3=BatchNormalization()(art_conv5_3)
    ac_art5_3=Activation("relu")(bn_art5_3)
    art_conv5_4 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=7)(t_atten5)
    bn_art5_4=BatchNormalization()(art_conv5_4)
    ac_art5_4=Activation("relu")(bn_art5_4)
    h5=concatenate(inputs=[ac_art5_1,ac_art5_2,ac_art5_3,ac_art5_4],axis=3)

    art_conv4_1 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=1)(t_atten4)
    bn_art4_1=BatchNormalization()(art_conv4_1)
    ac_art4_1=Activation("relu")(bn_art4_1)
    art_conv4_2 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=3)(t_atten4)
    bn_art4_2=BatchNormalization()(art_conv4_2)
    ac_art4_2=Activation("relu")(bn_art4_2)
    art_conv4_3 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=5)(t_atten4)
    bn_art4_3=BatchNormalization()(art_conv4_3)
    ac_art4_3=Activation("relu")(bn_art4_3)
    art_conv4_4 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=7)(t_atten4)
    bn_art4_4=BatchNormalization()(art_conv4_4)
    ac_art4_4=Activation("relu")(bn_art4_4)
    h4 = concatenate(inputs=[ac_art4_1, ac_art4_2, ac_art4_3, ac_art4_4], axis=3)

    art_conv3_1 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=1)(t_atten3)
    bn_art3_1=BatchNormalization()(art_conv3_1)
    ac_art3_1=Activation("relu")(bn_art3_1)
    art_conv3_2 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=3)(t_atten3)
    bn_art3_2=BatchNormalization()(art_conv3_2)
    ac_art3_2=Activation("relu")(bn_art3_2)
    art_conv3_3 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=5)(t_atten3)
    bn_art3_3=BatchNormalization()(art_conv3_3)
    ac_art3_3=Activation("relu")(bn_art3_3)
    art_conv3_4 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=7)(t_atten3)
    bn_art3_4=BatchNormalization()(art_conv3_4)
    ac_art3_4=Activation("relu")(bn_art3_4)
    h3 = concatenate(inputs=[ac_art3_1, ac_art3_2, ac_art3_3, ac_art3_4], axis=3)

    art_conv2_1 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=1)(t_atten2)
    bn_art2_1=BatchNormalization()(art_conv2_1)
    ac_art2_1=Activation("relu")(bn_art2_1)
    art_conv2_2 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=3)(t_atten2)
    bn_art2_2=BatchNormalization()(art_conv2_2)
    ac_art2_2=Activation("relu")(bn_art2_2)
    art_conv2_3 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=5)(t_atten2)
    bn_art2_3=BatchNormalization()(art_conv2_3)
    ac_art2_3=Activation("relu")(bn_art2_3)
    art_conv2_4 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=7)(t_atten2)
    bn_art2_4=BatchNormalization()(art_conv2_4)
    ac_art2_4=Activation("relu")(bn_art2_4)
    h2 = concatenate(inputs=[ac_art2_1, ac_art2_2, ac_art2_3, ac_art2_4], axis=3)

    art_conv1_1 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=1)(t_atten1)
    bn_art1_1=BatchNormalization()(art_conv1_1)
    ac_art1_1=Activation("relu")(bn_art1_1)
    art_conv1_2 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=3)(t_atten1)
    bn_art1_2=BatchNormalization()(art_conv1_2)
    ac_art1_2=Activation("relu")(bn_art1_2)
    art_conv1_3 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=5)(t_atten1)
    bn_art1_3=BatchNormalization()(art_conv1_3)
    ac_art1_3=Activation("relu")(bn_art1_3)
    art_conv1_4 = Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=glorot_normal(),
                         dilation_rate=7)(t_atten1)
    bn_art1_4=BatchNormalization()(art_conv1_4)
    ac_art1_4=Activation("relu")(bn_art1_4)
    h1 = concatenate(inputs=[ac_art1_1, ac_art1_2, ac_art1_3, ac_art1_4], axis=3)


    s5=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal(),activation="relu")(h5)
    s4=Add()([Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(h4),UpSampling2D(size=2)(s5)])
    s3=Add()([Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(h3),UpSampling2D(size=2)(s4)])
    s2=Add()([Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(h2),UpSampling2D(size=2)(s3)])
    s1=Add()([Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(h1),UpSampling2D(size=2)(s2)])
    sm1=UpSampling2D(size=(1,1))(s1)
    sm2=UpSampling2D(size=(2,2))(s2)
    sm3=UpSampling2D(size=(4,4))(s3)
    sm4=UpSampling2D(size=(8,8))(s4)
    sm5=UpSampling2D(size=(16,16))(s5)
    s=concatenate([sm1,sm2,sm3,sm4,sm5],axis=-1)
    fcm1=Conv2D(filters=32,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(s)
    bn_fm1=BatchNormalization()(fcm1)
    ac_fm1=Activation("relu")(bn_fm1)
    fcm2=Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_fm1)
    bn_fm2=BatchNormalization()(fcm2)
    ac_fm2=Activation("relu")(bn_fm2)
    fcm3=Conv2D(filters=1,kernel_size=3,padding="same",kernel_initializer=glorot_normal())(ac_fm2)
    bn_fm3=BatchNormalization()(fcm3)
    ac_fm3=Activation("sigmoid")(bn_fm3)

    model=Model(inputs=inputs,outputs=ac_fm3)
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy",metrics=["acc",metrics.MAE,fbeta_score])
    model.load_weights("weights.48.h5",by_name=True)
    return model
def dice_loss(y_true,y_pred):
    eps=1e-5
    intersectioin=K.sum(y_true*y_pred)
    union=K.sum(y_true)+K.sum(y_pred)+eps
    loss=1.-(2*intersectioin/union)
    return loss
def dice_loss(y_true,y_pred):
    eps=1e-5
    intersectioin=K.sum(y_true*y_pred)
    union=K.sum(y_true)+K.sum(y_pred)+eps
    loss=1.-(2*intersectioin/union)
    return loss


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = 0.3
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

if __name__=="__main__":
    model=get_model()
    model.summary()











