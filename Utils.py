from cgi import test
import scipy.io
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import os

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
    #gt_out = []
    groundtruth =[]    
    CSI = data[0] # (5000, 1)
    Pilots = data[1] 
    Phypayload = data[7] # double
    Groundtruth = data[4]
    #temp = Phypayload[1][0].reshape(40,48,1)
    #temp1 = temp.reshape(40,48,1)
    #print(Phypayload[1][0][0].reshape(40,48,1).shape)
    #print(temp[0])
    num_samples = CSI.shape[0]
    # mapping = np.array([1, 3,-3,-1])
    #mapping = np.array([-3, 3, -1, 1])
    #temp2= Groundtruth[1][0].reshape(40, 48, 1)
    #print(temp2[0])
    for i_sample in range(num_samples):
        #csi_out.append(np.concatenate((np.real(CSI[i_sample][0]).reshape(64, 1), np.imag(CSI[i_sample][0]).reshape(64, 1)), axis=-1))
        #pilot_out.append(np.concatenate((np.real(Pilots[i_sample][0]).reshape(40,4,1), np.imag(Pilots[i_sample][0]).reshape(40,4, 1)), axis=-1))
        csi_angle = np.real(CSI[i_sample][0].reshape(1, 48,1))
        csi_amp = np.imag(CSI[i_sample][0].reshape(1,48,1))        
        csi_out.append(np.concatenate((csi_amp,csi_angle),axis = 2))
            

        #csi_out = csi_out.reshape(1,48,2)
        pilot_angle = np.real(Pilots[i_sample][0].reshape(40, 4,1))  
        pilot_amp = np.imag(Pilots[i_sample][0].reshape(40, 4,1)) 
        pilot_out.append(np.concatenate((pilot_amp,pilot_angle),axis = 2))       
        #pilot_out.append([pilot_amp,pilot_angle])  
       
        phy_payload_angle = np.real(Phypayload[i_sample][0].reshape(40,48,1)) 
        phy_payload_amp = np.imag(Phypayload[i_sample][0].reshape(40,48,1))
        phy_payload.append((np.concatenate((phy_payload_amp,phy_payload_angle),axis = 2)))      
        #phy_payload.append([phy_payload_amp,phy_payload_angle])
        #groundtruth.append(np.transpose(mapping[np.intc(Groundtruth[i_sample][0])]).reshape(40, 48, 1))
        groundtruth_angle = np.real(Groundtruth[i_sample][0].reshape(40,48,1))
        groundtruth_amp = np.imag(Groundtruth[i_sample][0].reshape(40,48,1))
        groundtruth.append((np.concatenate((groundtruth_amp,groundtruth_angle),axis = 2)))
        #groundtruth.append([groundtruth_amp,groundtruth_angle]) 
    csi_out = np.array(csi_out)# (2, 48, 1)
    pilot_out = np.array(pilot_out) # (2, 40, 4)
    phy_payload = np.array(phy_payload) # (2, 40, 48)
    groundtruth = np.array(groundtruth) # (2, 40, 48)
    
    print('CSI_SHAPE=',csi_out.shape)
    print('pilot_SHAPE=',pilot_out.shape)
    print('phy_SHAPE=',phy_payload.shape)
    print('ground_SHAPE=',groundtruth.shape)

    return csi_out, pilot_out, phy_payload, groundtruth

def get_processed_dataset(data_path, split=4/5):
    file_list = os.listdir(data_path)
    CSI = np.empty((0, 1, 48, 2))
    PILOT = np.empty((0, 40, 4, 2))
    PHY_PAYLOAD = np.empty((0, 40, 48, 2))
    GROUNDTRUTH = np.empty((0, 40, 48, 2))
    #GT = np.empty((0, 40, 48, 1))
    file_list.sort()
    # print(file_list)
    for file in file_list:
       data_chunk = data_loader_for_each_payload(data_path + '/' + file)
       csi_out, pilot_out, phy_payload, groudtruth = data_preprocessing_for_each_payload(data_chunk)
       CSI = np.concatenate([CSI, csi_out], axis=0)
       PILOT = np.concatenate([PILOT, pilot_out], axis=0)
       PHY_PAYLOAD = np.concatenate([PHY_PAYLOAD, phy_payload], axis=0)
       GROUNDTRUTH = np.concatenate([GROUNDTRUTH, groudtruth], axis=0)
       #GT = np.concatenate([GT, gt], axis=0)
    
    num_samples = CSI.shape[0]
    rand_indices = np.random.permutation(num_samples)
    train_indices = rand_indices[:int(split*num_samples)]
    test_indices = rand_indices[int(split*num_samples):]
    
    #train_indices = np.random.permutation(range(5000, num_samples))
    
    #test_indices = list(range(0, 5000))

    np.savez_compressed("PHY_dataset_random_" + str(split), 
                        csi_train=CSI[train_indices, :, :, :],
                        pilot_train=PILOT[train_indices, :, :, :],
                        phy_payload_train=PHY_PAYLOAD[train_indices, :, :, :],
                        groundtruth_train=GROUNDTRUTH[train_indices, :, :, :],
                        csi_test=CSI[test_indices, :, :, :],
                        pilot_test=PILOT[test_indices, :, :, :],
                        phy_payload_test=PHY_PAYLOAD[test_indices, :, :, :],
                        groundtruth_test=GROUNDTRUTH[test_indices, :, :, :])
    print(num_samples)

def load_processed_dataset(path, shuffle_buffer_size, train_batch_size, test_batch_size):
    with np.load(path) as data:
        csi_train = data['csi_train'].astype(np.float32)
        pilot_train = data['pilot_train'].astype(np.float32)        
        phy_payload_train = data['phy_payload_train'].astype(np.float32)
        groundtruth_train = data['groundtruth_train'].astype(np.float32)

        csi_test = data['csi_test'].astype(np.float32)
        pilot_test = data['pilot_test'].astype(np.float32)       
        phy_payload_test = data['phy_payload_test'].astype(np.float32)
        groundtruth_test= data['groundtruth_test'].astype(np.float32)

    train_data = tf.data.Dataset.from_tensor_slices((csi_train, pilot_train,  phy_payload_train, groundtruth_train)).prefetch(1)
    train_data = train_data.shuffle(shuffle_buffer_size).batch(train_batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((csi_test, pilot_test,  phy_payload_test, groundtruth_test)).prefetch(1)
    test_data = test_data.batch(test_batch_size)
    
    
    x1 = np.multiply(phy_payload_test, groundtruth_test)>0
    x1 = np.multiply(x1[:, :, :, 0], x1[:, :, :, 1])
    print("baseline acc : ", np.mean(x1>0))

    return train_data, test_data

def NN_training(generator, discriminator, data_path, logdir):
    EPOCHS = 200
    runid = 'PHY_Net_x' + str(np.random.randint(10000))
    print(f"RUNID: {runid}")
    
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_mse = tf.keras.losses.CosineSimilarity(axis=2)   
    MSE_loss = tf.metrics.Mean()
    Accuracy = tf.metrics.Mean()
    G_loss = tf.metrics.Mean()
    D_loss = tf.metrics.Mean()
    
        
    train_data, test_data = load_processed_dataset(data_path, 500, 32, 32)
    print("The dataset has been loaded!")

    @tf.function
    def step(csi, pilot, phy_payload, groundtruth, training):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_out = generator(csi, pilot,phy_payload, training)
            
            d_real_logits = discriminator(groundtruth)
            d_fake_logits = discriminator(generated_out)
            d_loss_real = -tf.reduce_mean(d_real_logits)
            d_loss_fake = tf.reduce_mean(d_fake_logits)

            disc_loss = d_loss_real + d_loss_fake
            reconstruction_loss = loss_mse(groundtruth, generated_out)
            gen_loss = -d_loss_fake + reconstruction_loss

        if training:
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_weights))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_weights))
            for w in discriminator.trainable_variables:
                w.assign(tf.clip_by_value(w, -0.04, 0.04))
        #accuracy(gen_loss)
        G_loss(- d_loss_fake)
        D_loss(disc_loss)
        MSE_loss(reconstruction_loss)
        x1 = tf.cast(tf.math.multiply(tf.cast(groundtruth, tf.float32), tf.cast(generated_out, tf.float32)) > 0, tf.float32)
        Accuracy(tf.reduce_mean(tf.cast(tf.math.multiply(x1[:, :, :, 0], x1[:, :, :, 1]) > 0, tf.float32)))
        #Accuracy(tf.reduce_mean(tf.cast(tf.math.multiply(tf.cast(groundtruth, tf.float32), tf.cast(generated_out, tf.float32)) > 0, tf.float32)))
        return generated_out
    
    training_step = 0
    best_validation_acc = 0
    print("start training...")
    for epoch in range(EPOCHS):
        for csi, pilot, phy_payload, groundtruth in tqdm(train_data, desc=f'epoch {epoch+1}/{EPOCHS}', ascii=True):
            training_step += 1
            step(csi, pilot, phy_payload, groundtruth, training=True)

            if training_step % 200 == 0:
                with writer.as_default():
                    #print(f"c_loss: {c_loss:^6.3f} | acc: {acc:^6.3f}", end='\r')
                    tf.summary.scalar('train/d_loss', G_loss.result(), training_step)
                    tf.summary.scalar('train/g_loss', D_loss.result(), training_step)
                    tf.summary.scalar('train/mse_loss', MSE_loss.result(), training_step)
                    tf.summary.scalar('train/acc', Accuracy.result(), training_step)
                    G_loss.reset_states()
                    D_loss.reset_states()
                    MSE_loss.reset_states()
                    Accuracy.reset_states()
        G_loss.reset_states()
        D_loss.reset_states()
        MSE_loss.reset_states()
        Accuracy.reset_states()
        for csi, pilot,  phy_payload, label in test_data:
            generated_out = step(csi, pilot,  phy_payload, label, training=False)
            # print((generated_out.numpy())[0])

            with writer.as_default():
                tf.summary.scalar('test/d_loss', G_loss.result(), training_step)
                tf.summary.scalar('test/g_loss', D_loss.result(), training_step)
                tf.summary.scalar('test/mse_loss', MSE_loss.result(), training_step)
                tf.summary.scalar('test/acc', Accuracy.result(), training_step)
                if Accuracy.result() > best_validation_acc:
                    best_validation_acc = Accuracy.result()
                    generator.save_weights(os.path.join('saved_models', runid + '.tf'))
                G_loss.reset_states()
                D_loss.reset_states()
                MSE_loss.reset_states()
                Accuracy.reset_states()
if __name__ == "__main__":
    get_processed_dataset("dataset")
