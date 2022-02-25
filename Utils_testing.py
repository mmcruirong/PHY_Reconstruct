from cgi import test
import scipy.io
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import os
import time

def data_loader_for_each_payload(data_path):
    data = scipy.io.loadmat(data_path)
    data = data['data_set'][0][0]
    #label = int((data_path.split("."))[0].split("_")[1])
    return data

def normalize_data(x):
        return (x - np.mean(x)) / np.std(x)

def data_preprocessing_for_each_payload(data):
    csi_out = []
    pilot_out = []   
    phy_payload = []
    #raw_payload = []
    label = []
    label1 = []
    ber = []
    snr = []
    #gt_out = []
    groundtruth =[]    
    CSI = data[0] # (5000, 1)
    Pilots = data[1] 
    Phypayload = data[3] # Constellation -> Rx after EQ RAW -> Raw signal
    Groundtruth = data[2]
    #Raw_payload = data[4]
    Label = data[5]
    Label1 = data[6]
    BER = data[7]
    SNR = data[9]
    #mapping = np.array([0,1,2,3])
    #temp = mapping[Groundtruth[1][0]]
    #temp1 = temp.reshape(40,48,1)
    #print(temp[0])
    #print(temp[0])
    num_samples = CSI.shape[0]
    for i_sample in range(num_samples):
        #csi_out.append(np.concatenate((np.real(CSI[i_sample][0]).reshape(64, 1), np.imag(CSI[i_sample][0]).reshape(64, 1)), axis=-1))
        #pilot_out.append(np.concatenate((np.real(Pilots[i_sample][0]).reshape(40,4,1), np.imag(Pilots[i_sample][0]).reshape(40,4, 1)), axis=-1))
        csi_real = np.abs(CSI[i_sample][0]).reshape(1, 48,1)
        csi_imag = np.angle(CSI[i_sample][0]).reshape(1, 48,1)   
        csi_out.append(np.concatenate((csi_real,csi_imag),axis = 2))
            

        #csi_out = csi_out.reshape(1,48,2)
        pilot_real = np.abs(Pilots[i_sample][0]).reshape(40, 4,1)
        pilot_imag = np.angle(Pilots[i_sample][0]).reshape(40, 4,1)
        pilot_out.append(np.concatenate((pilot_imag,pilot_real),axis = 2))       
        #pilot_out.append([pilot_amp,pilot_angle])  

        #raw_payload_real = np.real(Raw_payload[i_sample][0]).reshape(40, 48,1, order='F')
        #raw_payload_imag = np.imag(Raw_payload[i_sample][0]).reshape(40, 48,1, order='F')
        #raw_payload.append((np.concatenate((raw_payload_real,raw_payload_imag),axis = 2)))    
       
        phy_payload_real = np.real(Phypayload[i_sample][0]).reshape(40, 48,1, order='F')
        phy_payload_imag = np.imag(Phypayload[i_sample][0]).reshape(40, 48,1, order='F')   
        phy_payload.append((np.concatenate((phy_payload_real,phy_payload_imag),axis = 2)))


        #groundtruth.append(np.transpose(mapping[np.intc(Groundtruth[i_sample][0])]).reshape(40, 48, 1))
        #groundtruth_real = np.real(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F') # use this line other than BPSK
        #groundtruth_imag = np.imag(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F') # use this line other than BPSK
        groundtruth_real = np.divide(np.real(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F'),0.707)  #this is only for BPSK
        groundtruth_imag = np.zeros((40, 48,1)) # This is only for BPSK
        groundtruth.append((np.concatenate((groundtruth_real,groundtruth_imag),axis = 2)))

        label.append(Label[i_sample][0].reshape(40,48,1, order='F'))
        label1.append(Label1[i_sample][0].reshape(40,48,1, order='F'))

        ber.append(BER[i_sample][0].reshape(1,1))
        snr.append(SNR[i_sample][0].reshape(1,1))
        #groundtruth.append([groundtruth_amp,groundtruth_angle]) 
    csi_out = np.array(csi_out)# (2, 48, 1)
    pilot_out = np.array(pilot_out) # (2, 40, 4)
    phy_payload = np.array(phy_payload) # (2, 40, 48)
    groundtruth = np.array(groundtruth) # (2, 40, 48)
    label = np.array(label)
    label1 = np.array(label1)
    ber = np.array(ber)
    snr = np.array(snr)
    print('CSI_SHAPE=',csi_out.shape)
    print('pilot_SHAPE=',pilot_out.shape)
    print('phy_SHAPE=',phy_payload.shape)
    print('ground_SHAPE=',groundtruth.shape)
    print('label_SHAPE=',label.shape)
    print('snr_shape=',snr.shape)

    return phy_payload, groundtruth, label, label1,csi_out,pilot_out,ber,snr

def get_processed_dataset(data_path, split=4/5):
    file_list = os.listdir(data_path)
    CSI = np.empty((0, 1, 48, 2))
    PILOT = np.empty((0, 40, 4, 2))
    PHY_PAYLOAD = np.empty((0, 40, 48, 2))
    GROUNDTRUTH = np.empty((0, 40, 48, 2))
    LABEL = np.empty((0, 40, 48, 1))
    LABEL1 = np.empty((0, 40, 48, 1))
    BER = np.empty((0, 1,1))
    SNR = np.empty((0, 1,1))
    #GT = np.empty((0, 40, 48, 1))
    file_list.sort()
    # print(file_list)
    for file in file_list:
       data_chunk = data_loader_for_each_payload(data_path + '/' + file)
       phy_payload, groudtruth, Tx_label, Rx_label,csi_out,pilot_out,ber,snr = data_preprocessing_for_each_payload(data_chunk)
       CSI = np.concatenate([CSI, csi_out], axis=0)
       PILOT = np.concatenate([PILOT, pilot_out], axis=0)
       PHY_PAYLOAD = np.concatenate([PHY_PAYLOAD, phy_payload], axis=0)
       GROUNDTRUTH = np.concatenate([GROUNDTRUTH, groudtruth], axis=0)
       LABEL = np.concatenate([LABEL, Tx_label], axis=0)
       LABEL1 = np.concatenate([LABEL1, Rx_label], axis=0)
       BER = np.concatenate([BER, ber], axis=0)
       SNR = np.concatenate([SNR, snr], axis=0)

       #GT = np.concatenate([GT, gt], axis=0)
    
    num_samples = LABEL.shape[0]


    print('BER =', np.mean(BER))
    print('SNR =', np.mean(SNR))

    np.savez_compressed("PHY_dataset_BB64",  #MC = Microwave ZB = Zigbee BB = babymonitor # WN = whitenoise #OW = OtherWiFi
                        csi_test=CSI,
                        pilot_test=PILOT,
                        phy_payload_test=PHY_PAYLOAD,
                        groundtruth_test=GROUNDTRUTH,
                        label_test=LABEL,
                        label1_test=LABEL1,
                        snr_test = SNR)

    print(num_samples)

def load_processed_dataset(path,path1, shuffle_buffer_size, train_batch_size, test_batch_size):
    with np.load(path) as data:

        csi_test = data['csi_test'].astype(np.float32)
        pilot_test = data['pilot_test'].astype(np.float32)       
        phy_payload_test = data['phy_payload_test'].astype(np.float32)
        groundtruth_test= data['groundtruth_test'].astype(np.float32)
        label_test = data['label_test'].astype(np.float32)
        label1_test = data['label1_test'].astype(np.float32)
        snr_test = data['snr_test'].astype(np.float32)



    with np.load(path1) as data:
        csi_train1 = data['csi_train'].astype(np.float32)
        pilot_train1 = data['pilot_train'].astype(np.float32)        
        #phy_payload_train1 = data['phy_payload_train'].astype(np.float32)
        #groundtruth_train1 = data['groundtruth_train'].astype(np.float32)
        #label_train1 = data['label_train'].astype(np.float32)
        #label1_train1 = data['label1_train'].astype(np.float32)
   
        csi_test1 = data['csi_test'].astype(np.float32)
        pilot_test1 = data['pilot_test'].astype(np.float32)       
        #phy_payload_test1 = data['phy_payload_test'].astype(np.float32)
        #groundtruth_test1 = data['groundtruth_test'].astype(np.float32)
        #label_test1 = data['label_test'].astype(np.float32)
        #label1_test1 = data['label1_test'].astype(np.float32)
    csi_train1 = csi_train1[10000:80000,:,:,:]
    pilot_train1 = pilot_train1[10000:80000,:,:,:]
    csi_test1 = csi_train1[0:10000,:,:,:]    
    pilot_test1 = pilot_train1[0:10000,:,:,:]  
    #csi_test1 = csi_test1[1000:2000,:,:,:]        
    #csi_train1 = csi_train1[2000:3000,:,:,:]  
    #pilot_test1 = pilot_test1[1000:2000,:,:,:] 
    #pilot_train1 = pilot_train1[2000:3000,:,:,:] 
    #csi_complete = np.concatenate([csi_train1,csi_test1], axis=0)
    #pilot_complete = np.concatenate([pilot_train1,pilot_test1], axis=0)
    #phy_payload_complete = np.concatenate([phy_payload_train1,phy_payload_test1], axis=0)
    #groundtruth_complete = np.concatenate([groundtruth_train1,groundtruth_test1], axis=0)
    #label_complete = np.concatenate([label_train1,label_test1], axis=0)
    #label1_complete = np.concatenate([label1_train1,label1_test1], axis=0)
    
    test_data = tf.data.Dataset.from_tensor_slices((csi_test, pilot_test,phy_payload_test, groundtruth_test, label_test,label1_test,csi_test1, pilot_test1,snr_test))#.cache().prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(test_batch_size)
    
    #print('Test_data',phy_payload_test.shape)
    #x1 = np.multiply(phy_payload_test, groundtruth_test)   #QPSK
    #x1 = np.multiply(x1[:, :, :, 0], x1[:, :, :, 1])  #QPSK



    test_data_np = np.array(tf.cast(tf.squeeze(label_test,axis = 3),tf.uint8))
    test_data1_np = np.array(tf.cast(tf.squeeze(label1_test,axis = 3),tf.uint8))
    test_data_np_bin = np.unpackbits(test_data_np,axis =2).astype(int)
    test_data1_np_bin = np.unpackbits(test_data1_np,axis =2).astype(int)
    bit_error = np.sum(np.abs(test_data_np_bin - test_data1_np_bin))/(10000*40*48*np.log2(16))
    #print('X1 shape = ',x1.shape)
    #for i in range(100):
        #print("baseline acc : ", np.mean(x1[i,:,:]>0))
    print("Testing baseline acc : ", bit_error)
    #print("Training baseline acc : ", np.mean(x_train>0))
    #print("Testing baseline acc : ", np.mean(x_test>0))

    return test_data

def NN_Testing(generator,  test_path, test_path1, logdir):
    testing_model = generator
    Mod_order = 64
    batch_size = 100
    count = 0
    modulation = '64QAM'
    Interferece = 'Whitenoise/'  #Whitenoise #OtherWiFi #Zigbee #BabyMonitor  #Microwave
    if Mod_order ==2:    
        testing_model.load_weights(os.path.join('saved_models/BPSK', 'PHY_Net_x2279.tf'))
    elif Mod_order ==4:   
        testing_model.load_weights(os.path.join('saved_models/QPSK', 'PHY_Net_x9106.tf'))
    elif Mod_order ==16:
        testing_model.load_weights(os.path.join('saved_models/16QAM','PHY_Net_x8356.tf'))
    elif Mod_order ==64:
        testing_model.load_weights(os.path.join('saved_models/64QAM','PHY_Net_x9269.tf'))
    print('weights loaded')    
    test_data = load_processed_dataset(test_path, test_path1,5000, batch_size, batch_size)
    #start_time = time.time()
    for csi, pilot,phy_payload,groundtruth, label, label1,csi1, pilot1,snr in test_data:
        Csi_duplicate = tf.repeat(csi,40,axis=0)            
        Csi_input = tf.squeeze(tf.reshape(Csi_duplicate,[40*batch_size,48,1,2]),axis = 2)
        Pilot_input = tf.squeeze(tf.reshape(pilot,[40*batch_size,4,1,2]),axis = 2)
        PHY_input = tf.squeeze(tf.reshape(phy_payload,[40*batch_size,48,1,2]),axis = 2)
        Groundtruth_input = tf.squeeze(tf.reshape(groundtruth,[40*batch_size,48,1,2]),axis = 2)
        Label_input = tf.squeeze(tf.reshape(label,[40*batch_size,48,1,1]),axis = 2)
        Label1_input = tf.squeeze(tf.reshape(label1,[40*batch_size,48,1,1]),axis = 2)
        
        Csi_duplicate1 = tf.repeat(csi1,40,axis=0)       
        Csi_input1 = tf.squeeze(tf.reshape(Csi_duplicate1,[40*batch_size,48,1,2]),axis = 2)
        Pilot_input1 = tf.squeeze(tf.reshape(pilot1,[40*batch_size,4,1,2]),axis = 2)
        #print('SNR shape = ',snr.shape)
        generated_out = testing_model([Csi_input, Pilot_input,Csi_input1, Pilot_input1,PHY_input,Groundtruth_input])
        #tf.print('Gen_out = ',generated_out[1,1,:])
        #generated_out = generator([csi, pilot,csi1, pilot1,phy_payload,groundtruth])
        classification_result = tf.math.argmax(generated_out,axis = 2)
        #tf.print('Gen_out = ',classification_result)
        
        classifcation_np = np.array(tf.cast(classification_result,tf.uint8))
        label_np = np.array(tf.cast(tf.squeeze(Label_input,axis = 2),tf.uint8))
        label1_np = np.array(tf.cast(tf.squeeze(Label1_input,axis = 2),tf.uint8))
        classification_bin = np.unpackbits(classifcation_np,axis =1).astype(int)
        label_bin = np.unpackbits(label_np,axis =1).astype(int)
        bit_error = np.sum(np.abs(label_bin - classification_bin))/(batch_size*40*48*np.log2(Mod_order))
        sinr = np.array(snr)

        
        
        scipy.io.savemat('test_results/'+Interferece + modulation+'_Origin/data%d.mat'%count, {'data': classifcation_np})
        scipy.io.savemat('test_results/'+Interferece + modulation+'_Origin/label%d.mat'%count, {'label': label_np})
        scipy.io.savemat('test_results/'+Interferece + modulation+'_Origin/sinr%d.mat'%count, {'sinr': sinr})
        
       
        #print("Save mat")
        scipy.io.savemat('test_results/'+Interferece + modulation+'_After/data%d.mat'%count, {'data_origin': label1_np})
        scipy.io.savemat('test_results/'+Interferece + modulation+'_After/label%d.mat'%count, {'label_origin': label_np})
    
        count = count +1
        
    #print('Inferencing time for frames:', time.time() - start_time)

       
if __name__ == "__main__":
    get_processed_dataset("test_dataset/babymonitor/64QAM")
