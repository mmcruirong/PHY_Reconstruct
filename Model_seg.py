from numpy.core.numeric import False_, outer
import tensorflow as tf


def feature_extractor_csi():
    inp = tf.keras.Input(shape=(48,2))
    out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(48)(out)
    return tf.keras.Model(inputs=inp, outputs=out)
    

def feature_extractor_pilot():
    inp = tf.keras.Input(shape=(4, 2))
    out = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(48)(out)
    return tf.keras.Model(inputs=inp, outputs=out)



def feature_extractor_PHY_payload():
    inp = tf.keras.Input(shape=(40, 48, 1))
    out = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    return tf.keras.Model(inputs=inp, outputs=out)

def corrector(): 
    f_csi = tf.keras.Input(shape=(48))
    f_pilot = tf.keras.Input(shape=(48))
    f_phy = tf.keras.Input(shape=(48,2)) # (batch_size, timestep, 128)
    out = tf.math.multiply(f_phy, tf.expand_dims(f_csi,1))
    #print(tf.expand_dims(tf.expand_dims(f_csi,1).shape))
    out = tf.math.multiply(out, tf.expand_dims(f_pilot,1))
    return tf.keras.Model(inputs=[f_csi, f_pilot, f_phy], outputs=out)

def generator():
    def stochastic_block(net, N, d):
        net = tf.keras.layers.Dense(N * d)(net)
        net = tf.concat([net[:, i*d:(i+1)*d] +
                        (1e-6 + tf.nn.softplus(net[:, (i+1)*d:(i+2)*d])) *
                        tf.random.normal(tf.shape(net[:, i*d:(i+1)*d]), 0, 1, dtype=tf.float32)
                        for i in range(N) if not (i % (2*d))], axis=1)
        return net

    def stochastic_block_v1(net, N, d):
        net = tf.keras.layers.Dense(N * d)(net)
        net = tf.concat([net[:, i*d:(i+1)*d] +
                        (1e-6 + tf.nn.softplus(net[:, (i+1)*d:(i+2)*d])) *
                        tf.random.uniform(tf.shape(net[:, i*d:(i+1)*d]), 0, 1, dtype=tf.float32)
                        for i in range(N) if not (i % (2*d))], axis=1)
        return net  
    inp = tf.keras.Input(shape=(48, 1))#, activation='leaky_relu'
    out = tf.keras.layers.Conv1D(filters=8, kernel_size=3, strides=2, padding='same', use_bias=False)(inp)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(out)   # (None, 2, 5, 6)
    out = tf.keras.layers.Flatten()(out) # (None, 60)
    out = stochastic_block_v1(out, 12, 2) # (None, 60)
    out = tf.keras.layers.Reshape((6,1))(out)
    out = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)    
    out = tf.keras.layers.Conv1D(filters=2, kernel_size = 3, strides=1, padding='same', use_bias=True)(out)
    #out = tf.keras.layers.Reshape((1920,1,2))(out)
    out = out + inp
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=2, kernel_size=3, strides=1, padding='same', use_bias=True)(out)
    return tf.keras.Model(inputs=inp, outputs=out)



def discriminator():   
    gen_out = tf.keras.Input(shape=(48, 4))
    out = tf.keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', use_bias=False)(gen_out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    #out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(out)
    out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)
    out = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(out)

    out = tf.keras.layers.Dense(1)(out)
    return tf.keras.Model(inputs=gen_out, outputs=out)

class PHY_Reconstruction_Generator(tf.keras.Model):
    def __init__(self):
        super(PHY_Reconstruction_Generator, self).__init__()
        self.csi_branch = feature_extractor_csi()
        self.pilot_branch = feature_extractor_pilot()        
        self.phy_generator = generator()
        self.phy_lstm = tf.keras.layers.LSTM(1, return_sequences=True) # (40, 48)
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.fusion_layer_1 = tf.keras.layers.Dense(48, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        self.fusion_layer_2 = tf.keras.layers.Dense(48, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        #self.activation = tf.keras.layers.Activation('sigmoid')
        self.DeConv_net_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=48, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(48, 1)),            
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, strides=1, padding='same', use_bias=False,
                activation=tf.nn.leaky_relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=3, strides=1, padding='same', use_bias=False,
                activation=tf.nn.leaky_relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                filters=16, kernel_size=3, strides=1, padding='same', use_bias=False,
                activation=tf.nn.leaky_relu),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv1D(
                filters=2, kernel_size=3, strides=1, padding='same')]
            #tf.keras.layers.Reshape(target_shape=(1920, 1, 2))]
            )  
        #self.PHY_payload_branch = discriminator()
    def call(self, CSI, Pilot, PHY_Payload, training=False):#, CSI, Pilot, Freq, 
        csi_features = self.csi_branch(CSI, training=training)
        pilot_features = self.pilot_branch(Pilot, training=training)
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
        out = phy_payload_generator * estimation_correction +  PHY_Payload  ''' 
        joint_features = self.concat_layer([csi_features, pilot_features])
        joint_features = self.fusion_layer_1(joint_features)
        joint_features = self.fusion_layer_2(joint_features)
        estimation_correction = self.DeConv_net_2(joint_features) 
        PHY_Corrected = PHY_Payload * estimation_correction +  PHY_Payload
        #PHY_Payload_Real = PHY_Corrected[:,:,0]
        #PHY_Payload_IMAG = PHY_Corrected[:,:,1]
        #whole_seq_output_real = self.phy_lstm(PHY_Payload_Real)
        LSTM_PHY_Payload = self.phy_lstm(PHY_Corrected)
        #LSTM_PHY_Payload = tf.stack([whole_seq_output_real,whole_seq_output_imag],2)
        out = self.phy_generator(LSTM_PHY_Payload, training=training)   +  PHY_Payload       
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
        LSTM_PHY_Payload = self.phy_lstm(PHY_Payload)
        #LSTM_PHY_Payload = tf.stack([whole_seq_output_real,whole_seq_output_imag],2) #LSTM_PHY_Payload
        phy_payload_discriminator = self.phy_discriminator(LSTM_PHY_Payload, training=training)     
        return phy_payload_discriminator
"""               
class PHY_Reconstruction_Net_(tf.keras.Model):
    # CSI (None, 64, 1) -> Conv1D
    # Pilot (None, 40, 4, 1) -> Conv2DConv2DTranspose
    # Freq (None, 1) -> FC
    # PHY_Payload (None, 40, 48, 1) -> Conv2D
    def __init__(self, num_classes=10):
        super(PHY_Reconstruction_Net, self).__init__()
        # feature extractors
        self.csi_branch = feature_extractor_csi()
        self.pilot_branch = feature_extractor_pilot()
        self.freq_branch = feature_extractor_freq()
        self.PHY_payload_branch = feature_extractor_PHY_payload()
        # aggregation layers
        self.flatten_layer_1 = tf.keras.layers.Flatten()
        self.flatten_layer_2 = tf.keras.layers.Flatten()
        self.flatten_layer_3 = tf.keras.layers.Flatten()
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.fusion_layer_1 = tf.keras.layers.Dense(64, activation='relu')
        self.fusion_layer_2 = tf.keras.layers.Dense(64, activation='relu')
        self.fusion_layer_3 = tf.keras.layers.Dense(64, activation='relu')
        self.fusion_layer_4 = tf.keras.layers.Dense(64, activation='relu')
        self.refine_layer = tf.keras.layers.Conv2D(
                filters=1, kernel_size=3, strides=1, padding='same')
        self.DeConv_net_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=5*6*3, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(5, 6, 3)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strideground_truths=2, padding='same', use_bias=False,
                activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),#PHY_features
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', use_bias=False,
                activation=tf.nn.leaky_relu),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same')])
                
        self.DeConv_net_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=5*6*3, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(5, 6, 3)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same', use_bias=False,
                activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', use_bias=False,
                activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kern X = Activation('relu')(X)el_size=3, strides=2, padding='same', use_bias=False,
                activation=tf.nn.leaky_relu),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same')])
        
        
        
        # predictor
        #self.prediction_layer = tf.keras.layers.Dense(num_classes)
    
    def call(self, CSI, Pilot, Freq, PHY_Payload, training=False):
        csi_features = self.csi_branch(CSI, training=training)
        pilot_features = self.pilot_branch(Pilot, training=training)
        freq_features = self.freq_branch(Freq, training=training)
        phy_payload_features = self.PHY_payload_branch(PHY_Payload, training=training)
        
        csi_features = self.flatten_layer_1(csi_features)
        pilot_features = self.flatten_layer_2(pilot_features)
        
        
        phy_payload_features = self.flatten_layer_3(phy_payload_features)
        phy_payload_features = self.fusion_layer_3(phy_payload_features)
        phy_payload_features = self.fusion_layer_4(phy_payload_features)
        # joint_features = phy_payload_features
        joint_features = self.concat_layer([csi_features, pilot_features, freq_features])
        joint_features = self.fusion_layer_1(joint_features)
        joint_features = self.fusion_layer_2(joint_features)
        # joint_features = phy_payload_features * weights
        # out = self.prediction_layer(joint_features)
        self_correction = self.DeConv_net_1(phy_payload_features)
        estimation_correction = self.DeConv_net_2(joint_features)
        out = self_correction * estimation_correction * PHY_Payload
        return out
#class PHY_Reconstruction_Net_LSTM(tf.keras.Model):
    # CSI (None, 64, 1) -> Conv1D
    # Pilot (None, 40, 4, 1) -> Conv2DConv2DTranspose
    # Freq (None, 1) -> FC
    # PHY_Payload (None, 40, 48, 1) -> Conv2D
#    def __init__(self, num_classes=10):
#        super(PHY_Reconstruction_Net_LSTM, self).__init__()
        # feature extractors
#        self.csi_branch = feature_extractor_csi()
#        self.pilot_branch = feature_extractor_pilot()
#        self.freq_branch = feature_extractor_freq()
#        self.phy_lstm = tf.keras.layers.LSTM(48, return_sequences=True) # (None, 40, 48)
#        self.phy_corrector = corrector()
    
#    def call(self, CSI, Pilot, Freq, PHY_Payload, training=False):
#        csi_features = self.csi_branch(CSI, training=training)
#        pilot_features = self.pilot_branch(Pilot, training=training)
        
#        PHY_Payload = PHY_Payload / tf.constant(3.1415926/4)
#        PHY_features = self.phy_lstm(PHY_Payload)
        
#        out = self.phy_corrector([csi_features, pilot_features, PHY_features])
#        return out
"""


    