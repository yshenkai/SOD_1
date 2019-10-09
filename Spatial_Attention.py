import keras.backend as K
from keras.engine.topology import Layer

class Spatial_Attention(Layer):

    def build(self, input_shape):
        _,self.H,self.W,self.C=input_shape

        self.w = self.add_weight(name="SpatialAttention_w_s",shape=[self.C, 1],dtype="float32",initializer="orthogonal",
                                 trainable=True)
        self.b = self.add_weight(name="SpatialAttention_b_s",shape=[1],dtype="float32",initializer="zeros",
                                 trainable=True)
        self.trainable_weights=[self.w,self.b]
        super(Spatial_Attention,self).build(input_shape)
    def call(self, inputs, **kwargs):
        spatial_attention_fm=K.dot(K.reshape(inputs,[-1,self.C]),self.w)+self.b
        #spatial_attention_fm = tf.matmul(tf.reshape(inputs, [-1, self.C]), self.w) + self.b
        spatial_attention_fm=K.sigmoid(K.reshape(spatial_attention_fm,[-1,self.W*self.H]))
        #spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, self.W * self.H]))
        #         spatial_attention_fm = tf.clip_by_value(tf.nn.relu(tf.reshape(spatial_attention_fm,
        #                                                                       [-1, W * H])),
        #                                                 clip_value_min = 0,
        #                                                 clip_value_max = 1)
        attention=K.reshape(K.concatenate([spatial_attention_fm]*self.C,axis=1),[-1,self.H,self.W,self.C])
        #attention = tf.reshape(tf.concat([spatial_attention_fm] * self.C, axis=1), [-1, self.H, self.W, self.C])
        attended_fm = attention * inputs
        return attended_fm
    def compute_output_shape(self, input_shape):
        return input_shape

