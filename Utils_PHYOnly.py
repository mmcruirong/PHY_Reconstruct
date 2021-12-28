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
    ser = []
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
    SER = data[8]
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
        #phy_payload.append([phy_payload_amp,phy_payload_angle])
        #groundtruth.append(np.transpose(mapping[np.intc(Groundtruth[i_sample][0])]).reshape(40, 48, 1))
        groundtruth_real = np.real(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F')
        groundtruth_imag = np.imag(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F') # use this line other than BPSK
        #groundtruth_real = np.divide(np.real(Groundtruth[i_sample][0]).reshape(40, 48,1, order='F'),0.707)  #this is only for BPSK
        #groundtruth_imag = np.zeros((40, 48,1)) # This is only for BPSK
        groundtruth.append((np.concatenate((groundtruth_real,groundtruth_imag),axis = 2)))

        label.append(Label[i_sample][0].reshape(40,48,1, order='F'))
        label1.append(Label1[i_sample][0].reshape(40,48,1, order='F'))

        ber.append(BER[i_sample][0].reshape(1,1))
        ser.append(SER[i_sample][0].reshape(1,1))
        #groundtruth.append([groundtruth_amp,groundtruth_angle]) 
    csi_out = np.array(csi_out)# (2, 48, 1)
    pilot_out = np.array(pilot_out) # (2, 40, 4)
    phy_payload = np.array(phy_payload) # (2, 40, 48)
    groundtruth = np.array(groundtruth) # (2, 40, 48)
    label = np.array(label)
    label1 = np.array(label1)
    ber = np.array(ber)
    ser = np.array(ser)
    print('CSI_SHAPE=',csi_out.shape)
    print('pilot_SHAPE=',pilot_out.shape)
    print('phy_SHAPE=',phy_payload.shape)
    print('ground_SHAPE=',groundtruth.shape)
    print('label_SHAPE=',label.shape)
    print('ber_shape=',ber.shape)

    return phy_payload, groundtruth, label, label1,csi_out,pilot_out,ber,ser

def get_processed_dataset(data_path, split=4/5):
    file_list = os.listdir(data_path)
    CSI = np.empty((0, 1, 48, 2))
    PILOT = np.empty((0, 40, 4, 2))
    PHY_PAYLOAD = np.empty((0, 40, 48, 2))
    GROUNDTRUTH = np.empty((0, 40, 48, 2))
    LABEL = np.empty((0, 40, 48, 1))
    LABEL1 = np.empty((0, 40, 48, 1))
    BER = np.empty((0, 1,1))
    SER = np.empty((0, 1,1))
    #GT = np.empty((0, 40, 48, 1))
    file_list.sort()
    # print(file_list)
    for file in file_list:
       data_chunk = data_loader_for_each_payload(data_path + '/' + file)
       phy_payload, groudtruth, Tx_label, Rx_label,csi_out,pilot_out,ber,ser = data_preprocessing_for_each_payload(data_chunk)
       CSI = np.concatenate([CSI, csi_out], axis=0)
       PILOT = np.concatenate([PILOT, pilot_out], axis=0)
       PHY_PAYLOAD = np.concatenate([PHY_PAYLOAD, phy_payload], axis=0)
       GROUNDTRUTH = np.concatenate([GROUNDTRUTH, groudtruth], axis=0)
       LABEL = np.concatenate([LABEL, Tx_label], axis=0)
       LABEL1 = np.concatenate([LABEL1, Rx_label], axis=0)
       BER = np.concatenate([BER, ber], axis=0)
       SER = np.concatenate([SER, ser], axis=0)

       #GT = np.concatenate([GT, gt], axis=0)
    
    num_samples = LABEL.shape[0]
    rand_indices = np.random.permutation(num_samples)
    train_indices = rand_indices[:int(split*num_samples)]
    test_indices = rand_indices[int(split*num_samples):]
    print('BER =', np.mean(BER[test_indices, :, :]))
    print('SER =', np.mean(SER[test_indices, :, :]))

    np.savez_compressed("PHY_dataset_BPSKSEGfull1_" + str(split), 
                        csi_train=CSI[train_indices, :, :, :],
                        pilot_train=PILOT[train_indices, :, :, :],
                        phy_payload_train=PHY_PAYLOAD[train_indices, :, :, :],
                        groundtruth_train=GROUNDTRUTH[train_indices, :, :, :],
                        label_train=LABEL[train_indices, :, :, :],
                        label1_train=LABEL1[train_indices, :, :, :],

                        csi_test=CSI[test_indices, :, :, :],
                        pilot_test=PILOT[test_indices, :, :, :],
                        phy_payload_test=PHY_PAYLOAD[test_indices, :, :, :],
                        groundtruth_test=GROUNDTRUTH[test_indices, :, :, :],
                        label_test=LABEL[test_indices, :, :, :],
                        label1_test=LABEL1[test_indices, :, :, :])

    print(num_samples)

def load_processed_dataset(path, shuffle_buffer_size, train_batch_size, test_batch_size):
    with np.load(path) as data:
        csi_train = data['csi_train'].astype(np.float32)
        pilot_train = data['pilot_train'].astype(np.float32)        
        phy_payload_train = data['phy_payload_train'].astype(np.float32)
        groundtruth_train = data['groundtruth_train'].astype(np.float32)
        label_train = data['label_train'].astype(np.float32)
        label1_train = data['label1_train'].astype(np.float32)

        csi_train = csi_train[2000:3000,:,:,:]        
        pilot_train = pilot_train[2000:3000,:,:,:] 
        phy_payload_train = phy_payload_train[2000:3000,:,:,:] 
        groundtruth_train = groundtruth_train[2000:3000,:,:,:]
        label_train = label_train[2000:3000,:,:,:]
        label1_train = label1_train[2000:3000,:,:,:]
        mixed_train = np.concatenate([phy_payload_train[0:35000,:,:,:],groundtruth_train[35000:40000,:,:,:]], axis=0)
        label1_mixed_train = np.concatenate([label1_train[0:35000,:,:,:],label1_train[35000:40000,:,:,:]], axis=0)

        #print('PHY SHAPE = ',mixed_train.shape)
   


        csi_test = data['csi_test'].astype(np.float32)
        pilot_test = data['pilot_test'].astype(np.float32)       
        phy_payload_test = data['phy_payload_test'].astype(np.float32)
        groundtruth_test= data['groundtruth_test'].astype(np.float32)
        label_test = data['label_test'].astype(np.float32)
        label1_test = data['label1_test'].astype(np.float32)

        csi_test = csi_test[1000:1100,:,:,:]        
        pilot_test = pilot_test[1000:1100,:,:,:] 
        phy_payload_test = phy_payload_test[1000:1100,:,:,:] 
        groundtruth_test = groundtruth_test[1000:1100,:,:,:]
        label_test = label_test[1000:1100,:,:,:]
        label1_test = label1_test[1000:1100,:,:,:]

    train_data = tf.data.Dataset.from_tensor_slices((csi_train, pilot_train,phy_payload_train, groundtruth_train,label_train,label1_train))#.cache().prefetch(tf.data.AUTOTUNE)
    train_data = train_data.shuffle(shuffle_buffer_size).batch(train_batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((csi_test, pilot_test,phy_payload_test, groundtruth_test, label_test,label1_test))#.cache().prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(test_batch_size)
    
    #print('Test_data',phy_payload_test.shape)
    x1 = np.multiply(phy_payload_test, groundtruth_test)   #QPSK
    x1 = np.multiply(x1[:, :, :, 0], x1[:, :, :, 1])  #QPSK

    #x_test = np.multiply(phy_payload_test[:, :, :, 0], groundtruth_test[:, :, :, 0])   # This is for BPSK
    #x_train = np.multiply(phy_payload_train[:, :, :, 0], groundtruth_train[:, :, :, 0])   # This is for BPSK
   
    #print('X1 shape = ',x1.shape)
    #for i in range(100):
        #print("baseline acc : ", np.mean(x1[i,:,:]>0))
    print("Training baseline acc : ", np.mean(x1>0))
    #print("Training baseline acc : ", np.mean(x_train>0))
    #print("Testing baseline acc : ", np.mean(x_test>0))

    return train_data, test_data

def NN_training(generator, discriminator, data_path, logdir):
    EPOCHS = 800
    batch_size = 100
    runid = 'PHY_Net_x' + str(np.random.randint(10000))
    print(f"RUNID: {runid}")
    Mod_order = 4
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    loss_binentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_cosine = tf.keras.losses.CosineSimilarity(axis=2,reduction=tf.keras.losses.Reduction.NONE)
    loss_mse = tf.keras.losses.MeanAbsoluteError()
    MSE_loss = tf.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()#tf.keras.metrics.MeanAbsoluteError()#tf.metrics.Mean()#
    G_loss = tf.metrics.Mean()
    D_loss = tf.metrics.Mean()
    batch_accuracy = 0
    testing_accuracy = 0
    train_data, test_data = load_processed_dataset(data_path, 500, batch_size, batch_size)
    print("The dataset has been loaded!")
    
       


    @tf.function
    def step(csi, pilot,phy_payload, groundtruth, label,label1, training):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #generated_out = generator(phy_payload, training)
            generated_out = generator([csi, pilot,phy_payload,groundtruth])

            print(generated_out.shape)
            print(label.shape)
            #d_real_logits = discriminator(groundtruth)            
            #d_fake_logits = discriminator(generated_out)
            #d_loss_real = loss_crossentropy(label,d_real_logits)
            #d_loss_fake = loss_crossentropy(tf.math.subtract(tf.math.multiply(tf.ones_like(label),3),label),d_fake_logits)
            #d_loss_real = loss_binentropy(tf.ones_like(d_real_logits),d_real_logits)
            #d_loss_fake = loss_binentropy(tf.zeros_like(d_fake_logits),d_fake_logits)
            #disc_loss = d_loss_real + d_loss_fake
            #reconstruction_loss = loss_cosine(groundtruth, generated_out)

            #generated_out = generator(label1,training)    
            #classficationloss = loss_crossentropy(label,generated_out)    
            gen_loss = loss_crossentropy(tf.reshape(label,[40*48*batch_size,1]),tf.reshape(generated_out,[40*48*batch_size,Mod_order])) #+ reconstruction_loss
            #gen_loss = loss_crossentropy(label,generated_out)
            #gen_loss = loss_mse(groundtruth,generated_out)

            #gen_loss = loss_cosine(groundtruth,generated_out)
        if training:
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
            #disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_weights))
            #discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_weights))
            #for w in discriminator.trainable_variables:
                #w.assign(tf.clip_by_value(w, -0.04, 0.04))
        accuracy(tf.reshape(label,[40*48*batch_size,1]),tf.reshape(generated_out,[40*48*batch_size,Mod_order]))
        #accuracy(groundtruth,generated_out)
        G_loss(gen_loss)
        #D_loss(disc_loss)
        #x1 = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.round(tf.cast(generated_out, tf.float32)), tf.cast(label, tf.float32)), tf.float32))


        #accuracy(tf.reduce_mean(tf.math.divide(tf.cast(x1, tf.float32),1006560)))
        #x1 = tf.cast(tf.math.multiply(tf.cast(groundtruth, tf.float32), tf.cast(generated_out, tf.float32)) > 0, tf.float32)
        #accuracy(tf.cast(tf.math.multiply(x1[:, :, 0], x1[:, :, 1]) > 0, tf.float32))
        #x1 = tf.cast(tf.math.multiply(tf.cast(groundtruth, tf.float32), tf.cast(generated_out, tf.float32)) > 0, tf.float32)
        #accuracy(tf.cast(tf.math.multiply(x1[:, :, 0], x1[:, :, 0]) > 0, tf.float32))
        #print(accuracy.result().shape)
        return generated_out
    
    
    training_step = 0
    testing_step = 0
    #best_validation_acc = 0
    print("start training...")
    for epoch in range(EPOCHS):
        for csi, pilot,phy_payload, groundtruth, label,label1 in tqdm(train_data, desc=f'epoch {epoch+1}/{EPOCHS}', ascii=True):
            # CSI (1,48,1,2) -> (40,48,2)
            # PILOT (1,40,4,2) -> (40,4,2)
            # PHY (1,1920,1,2) -> (40,48,2)
            # Groundtruth(1,1920,1,2)-> (40,48,2)
            # label (1,1920,1,1) -> (40,48,1)
            training_step += 1
            Csi_duplicate = tf.repeat(csi,40,axis=0)
            #tf.print('CSI_Duplicate',Csi_duplicate[1])
            #tf.print('CSI_Duplicate',Csi_duplicate[0])
            Csi_input = tf.squeeze(tf.reshape(Csi_duplicate,[40*batch_size,48,1,2]),axis = 2)
            #print('CSI_Input',Csi_input.shape)
            Pilot_input = tf.squeeze(tf.reshape(pilot,[40*batch_size,4,1,2]),axis = 2)
            PHY_input = tf.squeeze(tf.reshape(phy_payload,[40*batch_size,48,1,2]),axis = 2)
            Groundtruth_input = tf.squeeze(tf.reshape(groundtruth,[40*batch_size,48,1,2]),axis = 2)
            Label1_input = tf.squeeze(tf.reshape(label1,[40*batch_size,48,1,1]),axis = 2)
            Label_input = tf.squeeze(tf.reshape(label,[40*batch_size,48,1,1]),axis = 2)
            #print('CSI SHAPE = ',Csi_input.shape)
            #print('Pilot SHAPE = ',Pilot_input.shape)
            #print('PHY SHAPE = ',PHY_input.shape)
            #print('GT SHAPE = ',Groundtruth_input.shape)
            #print('label SHAPE = ',Label_input.shape)

            
            step(Csi_input, Pilot_input,PHY_input,Groundtruth_input, Label_input, Label1_input, training=True)
            batch_accuracy = accuracy.result() + batch_accuracy
            #print('batch_accuracy = ', batch_accuracy)
            if training_step % 100 == 0:
                with writer.as_default():
                    #print(f"c_loss: {c_loss:^6.3f} | acc: {acc:^6.3f}", end='\r')
                    tf.summary.scalar('train/g_loss', G_loss.result(), training_step)
                    tf.summary.scalar('train/acc', tf.divide(batch_accuracy,100), training_step)
                    G_loss.reset_states()                    
                    accuracy.reset_states()
                    batch_accuracy = 0         
                    #tf.print(tf.divide(batch_accuracy,100))
        G_loss.reset_states()
        accuracy.reset_states()
        #start_time = time.time()
        for csi, pilot,phy_payload,groundtruth, label, label1 in test_data:
            # same as training 
            Csi_duplicate = tf.repeat(csi,40,axis=0)            
            Csi_input = tf.squeeze(tf.reshape(Csi_duplicate,[40*batch_size,48,1,2]),axis = 2)
            Pilot_input = tf.squeeze(tf.reshape(pilot,[40*batch_size,4,1,2]),axis = 2)
            PHY_input = tf.squeeze(tf.reshape(phy_payload,[40*batch_size,48,1,2]),axis = 2)
            Groundtruth_input = tf.squeeze(tf.reshape(groundtruth,[40*batch_size,48,1,2]),axis = 2)
            Label_input = tf.squeeze(tf.reshape(label,[40*batch_size,48,1,1]),axis = 2)
            Label1_input = tf.squeeze(tf.reshape(label1,[40*batch_size,48,1,1]),axis = 2)

            testing_step += 1
            generated_out = step(Csi_input, Pilot_input,PHY_input,Groundtruth_input, Label_input,Label1_input, training=False)
            #tf.print('Gen_out = ',generated_out[1,1,:])
            classification_result = tf.math.argmax(generated_out,axis = 2)
            #tf.print('Gen_out = ',classification_result)
            total_bit_error = 0

            classification_bin = np.unpackbits(np.array(tf.cast(classification_result,tf.uint8)),axis =1).astype(int)
            label_bin = np.unpackbits(np.array(tf.cast(tf.squeeze(Label_input,axis = 2),tf.uint8)),axis =1).astype(int)
            bit_error = np.sum(np.abs(label_bin - classification_bin))
            #print(classification_result[0,0:4])
            #print(classification_bin[0,0:32])
            #print(tf.squeeze(Label_input,axis = 2)[0,0:4])
            #print(label_bin[0,0:32])
            #print('total', label_bin[0,0:32] - classification_bin[0,0:32])
            #for j in range(4000):
                #for i in range(48):           
                    #classification_bin = list(map(int,bin(int(classification_result[j,i]))[2:].zfill(4)))
                    #label_bin = list(map(int,bin(int(Label_input[j,i]))[2:].zfill(4)))
                    #print('classification = ', int(classification_result[j,i]))
                    #print('classification_bin = ', classification_bin)
                    #print('label = ', int(Label_input[j,i]))
                    #print('label_bin = ', label_bin)                   
                    #bit_error = np.sum(np.abs(np.array(classification_bin)-np.array(label_bin)))
                    #print('bit_error = ', bit_error)
                    #total_bit_error = total_bit_error + bit_error

            print('total_bit_error = ', bit_error/(4000*48))

            #tf.print('Gen_out = ',bin(int(classification_result)).replace("0b",""))
            #tf.print('label = ',tf.squeeze(Label_input,axis = 2))

            #difference = tf.math.abs(tf.math.subtract(tf.squeeze(tf.cast(Label_input, tf.float32),axis = 2), tf.cast(classification_result, tf.float32)))
            #tf.print('classification_result = ',tf.math.argmax(generated_out,axis = 2))
            #print('difference = ',difference)
            #tf.print('Average_BER = ', 1-tf.math.divide(tf.math.reduce_sum(difference),4000*48))     
            tf.print('Testing ACC = ',accuracy.result())
            testing_accuracy = accuracy.result() + testing_accuracy
            
            #print('batch_accuracy = ', testing_accuracy)
        #print('testing_step = ', testing_step)
            if testing_step % 100 == 0:
                with writer.as_default():
                #tf.summary.scalar('test/d_loss', D_loss.result(), testing_step)
                    tf.summary.scalar('test/g_loss', G_loss.result(), training_step)
                    tf.summary.scalar('test/acc', tf.divide(testing_accuracy,100), training_step)
                    tf.summary.scalar('train/d_loss',  tf.math.reduce_mean(testing_accuracy), training_step)

                    #if batch_accuracy.result() > best_validation_acc:
                        #best_validation_acc = batch_accuracy.result()
                        #generator.save_weights(os.path.join('saved_models', runid + '.tf'))
                    G_loss.reset_states()                                     
                    accuracy.reset_states()
                    #tf.print(tf.math.reduce_max(testing_accuracy))
                    testing_accuracy = 0  
        #print('Inferencing time for 10k frames:', time.time() - start_time)
if __name__ == "__main__":
    get_processed_dataset("BPSK_full1")
