from numpy.core.numeric import False_, outer
import tensorflow as tf
import numpy as np

scale = 2

def feature_extractor_csi():
    inp = tf.keras.Input(shape=(48,2))
    out = tf.keras.layers.Conv1D(filters=int(16*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)
    
def feature_extractor_csi1():
    inp = tf.keras.Input(shape=(48,2))
    out = tf.keras.layers.Conv1D(filters=int(16*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def feature_extractor_csi_comb():
    inp = tf.keras.Input(shape=(48,128*scale))
    out = tf.keras.layers.Conv1D(filters=int(256*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1D(filters=int(256*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def feature_extractor_pilot():
    inp = tf.keras.Input(shape=(4, 2))
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(96*scale)(out)
    out = tf.keras.layers.Reshape([6,16*scale])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(256*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(128*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def feature_extractor_pilot1():
    inp = tf.keras.Input(shape=(4, 2))
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(96*scale)(out)
    out = tf.keras.layers.Reshape([6,16*scale])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(256*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(128*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def feature_extractor_pilot_comb():
    inp = tf.keras.Input(shape=(48, 128*scale))
    out = tf.keras.layers.Conv1D(filters=int(256*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=2, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(int(128*scale))(out)
    return tf.keras.Model(inputs=inp, outputs=out)


def CrossCNN():
    inp = tf.keras.Input(shape=(48,512))#, activation='leaky_relu'
    out = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dense(256)(out)
    
    return tf.keras.Model(inputs=inp, outputs=out)

def CNN():
    inp = tf.keras.Input(shape=(48,300))#, activation='leaky_relu'
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(256*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(1536)(out)
    #mu, rho = tf.split(out, num_or_size_splits=2, axis=1)
    #sd = tf.math.log(1+tf.math.exp(rho))    
    #encoder_out = mu + sd * tf.random.normal([6*128], 0, 1, tf.float32)
    #decoder_inp = tf.keras.layers.Dense(6*128,activation = None)(encoder_out)
    out = tf.keras.layers.Reshape([12,128])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(128*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(64*scale), kernel_size=2, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1DTranspose(filters=int(32*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.ReLU()(out)
    #out = tf.keras.layers.Dense(2)(out)
    out = tf.keras.layers.Dropout(.25)(out)
    out = tf.keras.layers.Dense(16,activation = 'softmax')(out)
    
    return tf.keras.Model(inputs=inp, outputs=out)



""""
def discriminator():   
    gen_out = tf.keras.Input(shape=(40, 48,2))
    out = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=2, padding='same', use_bias=False)(gen_out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(out)
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='same', use_bias=False)(out)

    out = tf.keras.layers.Dense(40)(out)
    return tf.keras.Model(inputs=gen_out, outputs=out)

def scale_dot():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)

def scale_dot1():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)
def scale_dot2():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)
def scale_dot3():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)
def scale_dot4():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)
def scale_dot5():
    input1 = tf.keras.Input(shape=(48,64*scale))
    input2 = tf.keras.Input(shape=(48,64*scale))
    input3 = tf.keras.Input(shape=(48,64*scale))

    out = tf.matmul(input1, input2,transpose_b=True)
    #csi_branch+pilot_branch
    out = tf.math.divide(out,8*np.sqrt(scale))
    out = tf.keras.layers.Activation('tanh')(out)
    #out = tf.keras.layers.Softmax()(out)
    out = tf.matmul(out, input3)
    return tf.keras.Model(inputs=[input1,input2,input3], outputs=out)

def multiATT():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)

def multiATT1():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot1()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot1()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot1()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot1()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)
def multiATT2():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot2()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot2()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot2()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot2()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)
def multiATT3():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot3()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot3()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot3()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot3()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)
def multiATT4():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot4()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot4()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot4()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot4()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)
def multiATT5():
    input1 = tf.keras.Input(shape=(48,32*scale))
    input2 = tf.keras.Input(shape=(48,32*scale))
    input3 = tf.keras.Input(shape=(48,32*scale))

    Scale_input1_1 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_1 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_1 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_2 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_2 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_2 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_3 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_3 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_3 = tf.keras.layers.Dense(64*scale)(input3)

    Scale_input1_4 = tf.keras.layers.Dense(64*scale)(input1)
    Scale_input2_4 = tf.keras.layers.Dense(64*scale)(input2)
    Scale_input3_4 = tf.keras.layers.Dense(64*scale)(input3)


    csi1 = scale_dot5()([Scale_input1_1,Scale_input2_1,Scale_input3_1])
    csi2 = scale_dot5()([Scale_input1_2,Scale_input2_2,Scale_input3_2])
    csi3 = scale_dot5()([Scale_input1_3,Scale_input2_3,Scale_input3_3])
    csi4 = scale_dot5()([Scale_input1_4,Scale_input2_4,Scale_input3_4])  

    csi_concate = tf.concat([csi1,csi2,csi3,csi4],2)
    csi_out = tf.keras.layers.Dense(32*scale)(csi_concate)
    
    csi_out = csi_out + input3
    csi_ATTout =  tf.keras.layers.LayerNormalization()(csi_out)
   

    return tf.keras.Model(inputs=[input1,input2,input3], outputs=csi_ATTout)
def CSI_Pilot_Features():
    f_csi = tf.keras.Input(shape=(48,2))
    f_pilot = tf.keras.Input(shape=(4,2))
    f_csi1 = tf.keras.Input(shape=(48,2))
    f_pilot1 = tf.keras.Input(shape=(4,2))
    
    csi_branch = feature_extractor_csi()(f_csi)
    pilot_branch = feature_extractor_pilot()(f_pilot)

    csi_branch1 = feature_extractor_csi1()(f_csi1)
    pilot_branch1 = feature_extractor_pilot1()(f_pilot1)


    csi_att = multiATT()([csi_branch,csi_branch,csi_branch])
    csi_out = tf.keras.layers.Dense(32*scale)(csi_att)
    csi_out = csi_att + csi_out 
    csi_att = tf.keras.layers.LayerNormalization()(csi_out)

    pilot_att = multiATT()([pilot_branch,pilot_branch,pilot_branch])
    
    combined_att = multiATT()([csi_att,csi_att,pilot_att])


    csi_att1 = multiATT1()([csi_branch1,csi_branch1,csi_branch1])
    csi_out1 = tf.keras.layers.Dense(32*scale)(csi_att1)
    csi_out1 = csi_att1 + csi_out1 
    csi_att1 = tf.keras.layers.LayerNormalization()(csi_out1)

    pilot_att1 = multiATT1()([pilot_branch1,pilot_branch1,pilot_branch1])
    
    combined_att1 = multiATT1()([csi_att1,csi_att1,pilot_att1])

    out_att = tf.keras.layers.Dense(32)(combined_att)
    out_att1 = tf.keras.layers.Dense(32)(combined_att1)
    out = out_att + combined_att
    out1 = out_att1 + combined_att1
    out = tf.keras.layers.LayerNormalization()(out)
    out1 = tf.keras.layers.LayerNormalization()(out1)
    out = tf.keras.layers.Dense(64)(out)
    out1 = tf.keras.layers.Dense(64)(out1)

    #csi_branch = feature_extractor_csi()(f_csi)
    #pilot_branch = feature_extractor_pilot()(f_pilot)
    #encoder_out * csi_branch * pilot_branch
    #inp_concate1 = tf.concat([inp11,inp21,inp31,inp41],2)#encoder_out * csi_branch * pilot_branch
    
    
    #out1 = tf.keras.layers.Dense(64)(inp_concate1)
    Channel_Int = out * out1
    #EQ_out = tf.concat([inp,csi_branch,pilot_branch],2)
    #out = tf.keras.layers.Dense(32)(Channel_Int)
    #EQ_out = tf.keras.layers.Dense(2,activation = 'Softmax')(out)
    features = tf.keras.layers.Dense(128*scale)(Channel_Int)

    return tf.keras.Model(inputs=[f_csi,f_pilot,f_csi1,f_pilot1], outputs=features)

"""



def PHY_Reconstruction_AE():
    #EQ_in = tf.keras.Input(shape=(48,2))
    f_csi = tf.keras.Input(shape=(48,2))
    f_pilot = tf.keras.Input(shape=(4,2))
    f_csi1 = tf.keras.Input(shape=(48,2))
    f_pilot1 = tf.keras.Input(shape=(4,2))
    inp = tf.keras.Input((48,2))   
    ground_truth = tf.keras.Input((48,2))  
    
    csi_branch = feature_extractor_csi()(f_csi)
    pilot_branch = feature_extractor_pilot()(f_pilot)

    csi_branch1 = feature_extractor_csi1()(f_csi1)
    pilot_branch1 = feature_extractor_pilot1()(f_pilot1)
    
    

    phy_branch = tf.keras.layers.Dense(128*scale)(inp)
    #groundtruth_branch = tf.keras.layers.Dense(128)(ground_truth)

    
    CSI_diff = csi_branch - csi_branch1
    pilot_diff = pilot_branch - pilot_branch1

    #MultiAtt_out_csi = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=1)(CSI_diff,CSI_diff)
    #MultiAtt_out_pilot = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=1)(pilot_diff,pilot_diff)
    cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=1)(pilot_diff,CSI_diff)
    #cross_attention=  tf.keras.layers.LayerNormalization()(cross_attention)
    #print('Attentionshape =', MultiAtt_out_csi.shape)
    csi_feature_correct = feature_extractor_csi_comb()(CSI_diff)
    pilot_feature_correct = feature_extractor_pilot_comb()(pilot_diff)

    #print(mixed_feature.shape)
    #
    #features = CrossCNN()(mixed_feature)
    #features = CSI_Pilot_Features()([f_csi,f_pilot,f_csi1,f_pilot1])
    #channel_branch = tf.keras.layers.Dense(128)(features)
    #
    EQ_phy =  phy_branch - cross_attention -  csi_feature_correct - pilot_feature_correct #mixed_feature
    #Channel_features = tf.concat([EQ_phy,csi_feature_correct,pilot_feature_correct],2)
    EQ_out =  tf.keras.layers.Dense(int(512*scale))(EQ_phy)

    #print('Out_shape', features.shape)
    #phy_lstm_1 = tf.keras.layers.LSTMCell(int(128*scale), name='lstm1') # (40, 48)
    #correction = tf.keras.layers.LSTMCell(int(256*scale))
    #stackcell = [phy_lstm_1,correction]
    #LSTM_stackcell = tf.keras.layers.StackedRNNCells(stackcell)
    #Reconstructioncell = tf.keras.layers.RNN(LSTM_stackcell,return_state=True, return_sequences=True)
    encoder_out, state_h, state_c = tf.keras.layers.LSTM(300,return_state=True, return_sequences=True)(EQ_out) #Reconstructioncell(EQ_out)
    #out = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', use_bias=False)(ground_truth)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)    
    #out = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Dense(100)(out)  
    #encoder_lstm = tf.keras.layers.LSTM(64,return_state=True, return_sequences=True)
    #encoder_out, state_h, state_c = encoder_lstm(inp)
    #decoder_inp = encoder_out + out
    #decoder_lstm = tf.keras.layers.LSTM(64,return_state=True, return_sequences=True)
    #decoder_out,_,_, = decoder_lstm(decoder_inp,initial_state=[state_h, state_c])

   
 
    print('encoder_out', encoder_out.shape)

    out = CNN()(encoder_out) 
    
    #out = tf.keras.layers.Dense(2,activation = 'softmax')(encoder_out)  
    #out = tf.keras.layers.Dense(4,activation = 'softmax')(decoder_out)
    return tf.keras.Model(inputs=[f_csi,f_pilot,f_csi1,f_pilot1,inp,ground_truth], outputs=out)

