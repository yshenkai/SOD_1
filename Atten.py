from keras.engine.topology import Layer
import tensorflow as tf
import keras.backend as K
from keras.initializers import orthogonal
class Atten(Layer):
    def build(self,input_shape):
        _,self.H,self.W,self.C=input_shape
        self.w_c=self.add_weight("ChannelWiseAttention_w_c",shape=[self.C,self.C],dtype="float32",initializer="orthogonal",
                                 trainable=True)
        self.b_c = self.add_weight("ChannelWiseAttention_b_c",shape=[self.C],dtype="float32",initializer="zeros",
                                   trainable=True)
        self.w_s=self.add_weight("SpatialAttention_w_s",shape=[self.C, 1],dtype="float32",initializer="orthogonal",
                                 trainable=True)
        self.b_s=self.add_weight("SpatialAttention_b_s",shape=[1],dtype="float32",initializer="zeros",
                                 trainable=True)
        self.trainable_weights=[self.w_c,self.w_s,self.b_s,self.b_c]
        super(Atten,self).build(input_shape)
    def call(self, inputs, **kwargs):
        transpose_feature_map=K.permute_dimensions(K.mean(inputs,[1,2],keepdims=True),pattern=[0,3,1,2])
        channel_wise_attention_fm=K.dot(K.reshape(transpose_feature_map,[-1,self.C]),self.w_c)+self.b_c
        channel_wise_attention_fm=K.sigmoid(channel_wise_attention_fm)
        attention=K.reshape(K.concatenate([channel_wise_attention_fm]*(self.H*self.W),axis=1),[-1,self.H,self.W,self.C])
        channel_attention = attention * inputs

        spatial_attention_fm=K.dot(K.reshape(channel_attention,[-1,self.C]),self.w_s)+self.b_s
        spatial_attention_fm=K.sigmoid(K.reshape(spatial_attention_fm,[-1,self.W*self.H]))
        attention=K.reshape(K.concatenate([spatial_attention_fm]*self.C,axis=1),[-1,self.H,self.W,self.C])

        attended_fm = attention * channel_attention
        return attended_fm
    def compute_output_shape(self, input_shape):
        return input_shape


