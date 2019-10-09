from keras.engine.base_layer import Layer
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
import tensorflow as tf
class Channel_Wise_Attention(Layer):
    def build(self, input_shape):
        _,self.H,self.W,self.C=input_shape
        self.w=self.add_weight("ChannelWiseAttention_w_c",shape=[self.C, self.C],dtype="float32",
                               initializer="orthogonal",trainable=True)
        #self.w=tf.get_variable("ChannelWiseAttention_w_s", [self.C, self.C],
         #                     dtype=tf.float32,
         #                     initializer=tf.initializers.orthogonal)
        self.b=self.add_weight("ChannelWiseAttention_b_s",shape=[self.C],dtype="float32",initializer="zeros",
                               trainable=True)
       # self.b=tf.get_variable("ChannelWiseAttention_b_s", [self.C],
       #                       dtype=tf.float32,
        #                      initializer=tf.initializers.zeros)
        self.trainable_weights=[self.w,self.b]
        super(Channel_Wise_Attention,self).build(input_shape)
    def call(self, inputs, **kwargs):
        transpose_feature_map=K.permute_dimensions(K.mean(inputs,[1,2],keepdims=True),pattern=[0, 3, 1, 2])
        #transpose_feature_map = tf.transpose(tf.reduce_mean(inputs, [1, 2], keepdims=True),
         #                                    perm=[0, 3, 1, 2])

        channel_wise_attention_fm=K.dot(K.reshape(transpose_feature_map,[-1,self.C]),self.w)+self.b

        #channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
         #                                                [-1, self.C]),self.w) + self.b
        channel_wise_attention_fm=K.sigmoid(channel_wise_attention_fm)
        #channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
        attention=K.reshape(K.concatenate([channel_wise_attention_fm]*(self.H * self.W),axis=1),[-1,self.H, self.W, self.C])
        #attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (self.H * self.W),
        #                                 axis=1), [-1, self.H, self.W, self.C])
        attended_fm = attention * inputs
        return attended_fm

