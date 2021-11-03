from cgi import test
import scipy.io
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import os

def data_loader_for_each_payload(data_path):
    data = scipy.io.loadmat(data_path)
    data = data['data_set'][0][0]
    label = int((data_path.split("."))[0].split("_")[1])
    return data, label

def data_preprocessing_for_each_payload(data, label):
    csi_out = []
    pilot_out = []
    freq_out = []
    phy_payload = []
    label_out = []

    CSI = data[0] # (5000, 1)
    Pilots = data[1]
    freq = data[2] # 
    Phypayload = data[3] # double
    # print(Phypayload[2][0].shape)
    num_samples = CSI.shape[0]
    for i_sample in range(num_samples):
        csi_out.append(np.abs(CSI[i_sample][0].reshape(64, 1)))
        pilot_out.append(np.abs(Pilots[i_sample][0].reshape(40, 4, 1)))
        freq_out.append(freq[i_sample][0].reshape(1))
        phy_payload.append(Phypayload[i_sample][0].reshape(40, 48, 1))
        label_out.append(np.array(label).reshape(1))

    csi_out = np.array(csi_out)
    pilot_out = np.array(pilot_out)
    freq_out = np.array(freq_out)
    phy_payload = np.array(phy_payload)
    label_out = np.array(label_out)

    return csi_out, pilot_out, freq_out, phy_payload, label_out
    
def get_processed_dataset(path, split=4/5):
    file_list = os.listdir(path)
    
    CSI = np.empty((0, 64, 1))
    PILOT = np.empty((0, 40, 4, 1))
    FREQ = np.empty((0, 1))
    PHY_PAYLOAD = np.empty((0, 40, 48, 1))
    LABEL = np.empty((0, 1))

    for file in file_list:
       data_chunk, label = data_loader_for_each_payload(path + '/' + file)
       csi_out, pilot_out, freq_out, phy_payload, label = data_preprocessing_for_each_payload(data_chunk, label)
       CSI = np.concatenate([CSI, csi_out], axis=0)
       PILOT = np.concatenate([PILOT, pilot_out], axis=0)
       FREQ = np.concatenate([FREQ, freq_out], axis=0)
       PHY_PAYLOAD = np.concatenate([PHY_PAYLOAD, phy_payload], axis=0)
       LABEL = np.concatenate([LABEL, label], axis=0)
    
    num_samples = CSI.shape[0]
    rand_indices = np.random.permutation(num_samples)
    train_indices = rand_indices[int(split*num_samples):]
    test_indices = rand_indices[:int(split*num_samples)]

    np.savez_compressed("PHY_dataset_" + str(split), 
                        csi_train=CSI[train_indices, :, :],
                        pilot_train=PILOT[train_indices, :, :, :],
                        freq_train=FREQ[train_indices, :],
                        phy_payload_train=PHY_PAYLOAD[train_indices, :, :, :],
                        label_train=LABEL[train_indices, :],
                        csi_test=CSI[test_indices, :, :],
                        pilot_test=PILOT[test_indices, :, :, :],
                        freq_test=FREQ[test_indices, :],
                        phy_payload_test=PHY_PAYLOAD[test_indices, :, :, :],
                        label_test=LABEL[test_indices, :])


def load_processed_dataset(path, shuffle_buffer_size, train_batch_size, test_batch_size):
    with np.load(path) as data:
        csi_train = data['csi_train'].astype(np.float32)
        pilot_train = data['pilot_train'].astype(np.float32)
        freq_train = data['freq_train'].astype(np.float32)
        phy_payload_train = data['phy_payload_train'].astype(np.float32)
        label_train = data['label_train'].astype(np.int32)

        csi_test = data['csi_test'].astype(np.float32)
        pilot_test = data['pilot_test'].astype(np.float32)
        freq_test = data['freq_test'].astype(np.float32)
        phy_payload_test = data['phy_payload_test'].astype(np.float32)
        label_test = data['label_test'].astype(np.int32)

    train_data = tf.data.Dataset.from_tensor_slices((csi_train, pilot_train, freq_train, phy_payload_train, label_train))
    train_data = train_data.shuffle(shuffle_buffer_size).batch(train_batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((csi_test, pilot_test, freq_test, phy_payload_test, label_test))
    test_data = test_data.batch(test_batch_size)

    return train_data, test_data

def NN_training(model, data_path, logdir):
    EPOCHS = 200
    runid = 'PHY_Net_x' + str(np.random.randint(10000))
    print(f"RUNID: {runid}")
    
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    optimizer = tf.optimizers.Adam(1e-4)
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()

    train_data, test_data = load_processed_dataset(data_path, 500, 64, 256)
    print("The dataset has been loaded!")

    @tf.function
    def step(csi, pilot, freq, phy_payload, label, training):

        with tf.GradientTape() as tape:
            outs = model(csi, pilot, freq, phy_payload, training)
            loss = loss_fn(label, outs)
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        accuracy(label, outs)
        cls_loss(loss)
    
    training_step = 0
    best_validation_acc = 0
    print("start training...")
    for epoch in range(EPOCHS):
        for csi, pilot, freq, phy_payload, label in tqdm(train_data, desc=f'epoch {epoch+1}/{EPOCHS}', ascii=True):
            training_step += 1
            step(csi, pilot, freq, phy_payload, label, training=True)

            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    print(f"c_loss: {c_loss:^6.3f} | acc: {acc:^6.3f}", end='\r')
                    tf.summary.scalar('train/acc', acc, training_step)
                    tf.summary.scalar('train/loss', c_loss, training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
        
        cls_loss.reset_states()
        accuracy.reset_states()

        for csi, pilot, freq, phy_payload, label in test_data:
            step(csi, pilot, freq, phy_payload, label, training=False)

            with writer.as_default():
                tf.summary.scalar('test/acc', accuracy.result(), training_step)
                tf.summary.scalar('test/loss', cls_loss.result(), training_step)
                if accuracy.result() > best_validation_acc:
                    best_validation_acc = accuracy.result()
                    model.save_weights(os.path.join('saved_models', runid + '.tf'))
                cls_loss.reset_states()
                accuracy.reset_states()


# get_processed_dataset("dataset")
