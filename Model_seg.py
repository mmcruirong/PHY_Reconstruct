from numpy.core.numeric import False_, outer
import tensorflow as tf
import numpy as np

scale = 1

def feature_extractor_csi():
    inp = tf.keras.Input(shape=(48,2))
    out = tf.keras.layers.Conv1D(filters=int(8*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(16*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(int(128*scale),activation = 'tanh')(out)
    return tf.keras.Model(inputs=inp, outputs=out)
    
def feature_extractor_csi1():
    inp = tf.keras.Input(shape=(48,2))
    out = tf.keras.layers.Conv1D(filters=int(8*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(16*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(int(128*scale),activation = 'tanh')(out)
    return tf.keras.Model(inputs=inp, outputs=out)


def feature_extractor_pilot():
    inp = tf.keras.Input(shape=(4, 2))
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(96)(out)
    out = tf.keras.layers.Reshape([6,16])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(32*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(16*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(8*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dense(int(128*scale),activation = 'tanh')(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def feature_extractor_pilot1():
    inp = tf.keras.Input(shape=(4, 2))
    out = tf.keras.layers.Conv1D(filters=int(32*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(96)(out)
    out = tf.keras.layers.Reshape([6,16])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(32*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(16*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(8*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Dense(int(128*scale),activation = 'tanh')(out)
    return tf.keras.Model(inputs=inp, outputs=out)


def CrossCNN():
    inp = tf.keras.Input(shape=(48,128))#, activation='leaky_relu'
    out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dense(128)(out)    
    out = tf.keras.layers.Flatten()(out) # (None, 60)
    out = tf.keras.layers.Reshape((6,128))(out)
    #out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(out)   # (None, 2, 5, 6)
    out = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)    
    #out = tf.keras.layers.Conv2D(filters=2, kernel_size = (3,3), strides=1, padding='same', use_bias=True)(out)
    #out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.Reshape((1920,1,2))(out)
    #out = out +inp
    
    #out = tf.keras.layers.Conv1D(filters=2, kernel_size=3, strides=1, padding='same', use_bias=True)(out)
    out = tf.keras.layers.Dense(int(128*scale),activation = 'tanh')(out)

    return tf.keras.Model(inputs=inp, outputs=out)

def CNN():
    inp = tf.keras.Input(shape=(48,400))#, activation='leaky_relu'
    out = tf.keras.layers.Conv1D(filters=int(64*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=int(128*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(filters=int(256*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(1536)(out)
    out = tf.keras.layers.Reshape([12,128])(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(128*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(64*scale), kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1DTranspose(filters=int(32*scale), kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    #out = tf.keras.layers.Dense(2)(out)
    out = tf.keras.layers.Dropout(.35)(out)
    out = tf.keras.layers.Dense(2,activation = 'softmax')(out)
    
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
    
    phy_branch = tf.keras.layers.Dense(128)(inp)
    
    CSI_diff = csi_branch - csi_branch1
    pilot_diff = pilot_branch - pilot_branch1

    MultiAtt_out_csi = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2)(CSI_diff,CSI_diff)
    MultiAtt_out_pilot = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2)(pilot_diff,pilot_diff)
    cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2)(MultiAtt_out_csi,MultiAtt_out_pilot)

    #print(MultiAtt_out_csi.shape)
    
    mixed_feature =  cross_attention*phy_branch + csi_branch + pilot_branch
    #print(mixed_feature.shape)

    features = CrossCNN()(mixed_feature)
    #features = CSI_Pilot_Features()([f_csi,f_pilot,f_csi1,f_pilot1])
    #channel_branch = tf.keras.layers.Dense(128)(features)
    #
    EQ_phy =  phy_branch  + features

    phy_lstm_1 = tf.keras.layers.LSTMCell(int(128*scale), name='lstm1') # (40, 48)
    correction = tf.keras.layers.LSTMCell(int(256*scale))
    stackcell = [phy_lstm_1,correction]
    LSTM_stackcell = tf.keras.layers.StackedRNNCells(stackcell)

    Reconstructioncell = tf.keras.layers.RNN(LSTM_stackcell,return_state=True, return_sequences=True)
    encoder_out, state_h, state_c = tf.keras.layers.LSTM(400,return_state=True, return_sequences=True)(EQ_phy) #Reconstructioncell(EQ_out)
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

    print('EQ_out_shape', phy_branch.shape)
 
    
    out = CNN()(encoder_out)
    #out = tf.keras.layers.Dense(4,activation = 'softmax')(decoder_out)
    return tf.keras.Model(inputs=[f_csi,f_pilot,f_csi1,f_pilot1,inp,ground_truth], outputs=out)

class CVAE(tf.keras.Model):

    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(40, 48, 2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(10 + 10),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(10,)),
                tf.keras.layers.Dense(units=5*6*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(5, 6, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    
    def call(self, PHY_Payload):
        mean, logvar = self.encode(PHY_Payload)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z,apply_sigmoid=True)
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        #logpz = log_normal_pdf(z, 0., 0.)
        #logqz_x = log_normal_pdf(z, mean, logvar)
        return x_logit

class PHY_Reconstruction_Generator(tf.keras.Model):
    def __init__(self):
        super(PHY_Reconstruction_Generator, self).__init__()  
        self.csi_branch = feature_extractor_csi()
        self.pilot_branch = feature_extractor_pilot()           
        self.phy_generator = generator()
        self.phy_lstm = tf.keras.layers.LSTM(1, return_sequences=True) # (40, 48)    
        #self.activation = tf.keras.layers.Activation('sigmoid')
      
        #self.PHY_payload_branch = discriminator()
    def call(self,csi, pilot, PHY_Payload, training=False):#, CSI, Pilot, Freq, 
       
        #PHY_Payload = PHY_Payload / tf.constant(3.1415926/4)
        '''PHY_Payload_Real = PHY_Payload[:,:,:,0]
        PHY_Payload_IMAG = PHY_Payload[:,:,:,1]
        whole_seq_output_real = self.phy_lstm(PHY_Payload_Real)
        whole_seq_output_imag = self.phy_lstm(PHY_Payload_IMAG)
        LSTM_PHY_Payload = tf.stack([whole_seq_output_real,whole_seq_output_imag],3)
        phy_payload_generator = self.phy_generator(LSTM_PHY_Payload, training=training)   
        #LSTM_PHY_Payload     
        joint_features = self.concat_layer([csi_features, pilot_features])
        joint_features = self.fusion_layer_1(joint_features)
        joint_features = self.fusion_layer_2(joint_features)
        estimation_correction = self.DeConv_net_2(joint_features)        
        out = phy_payload_generator * estimation_correction +  PHY_Payload'''  
        
        #LSTM_PHY_Payload = self.phy_lstm(PHY_Payload, training=training)
        #LSTM_PHY_Payload = tf.stack([whole_seq_output_real,whole_seq_output_imag],2)
        out = self.phy_generator(PHY_Payload, training=training)    
        #LSTM_PHY_Payload    
        #out = self.phy_generator(out, training=training)     
        #out =  self.activation(out)
        #out = self_correction    * estimation_correction  
        return out

class PHY_Reconstruction_discriminator(tf.keras.Model):
    def __init__(self):
        super(PHY_Reconstruction_discriminator, self).__init__()
        self.phy_discriminator = discriminator()
        self.phy_lstm = tf.keras.layers.LSTM(4, return_sequences=True) # (None, 40, 48)
    def call(self, PHY_Payload, training=False):
        #PHY_Payload = PHY_Payload / tf.constant(3.1415926/4)
        #PHY_Payload_Real = PHY_Payload
        #PHY_Payload_IMAG = PHY_Payload[:,:,1]
        #whole_seq_output_real = self.phy_lstm(PHY_Payload_Real)
        #LSTM_PHY_Payload = self.phy_lstm(PHY_Payload)
        #LSTM_PHY_Payload = tf.stack([whole_seq_output_real,whole_seq_output_imag],2) #LSTM_PHY_Payload
        phy_payload_discriminator = self.phy_discriminator(PHY_Payload, training=training)     
        return phy_payload_discriminator