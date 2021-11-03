from Model import PHY_Reconstruction_Net_LSTM
from Utils_v2 import NN_training

if __name__ == "__main__":
    PHY_Net = PHY_Reconstruction_Net_LSTM(num_classes=10)
    NN_training(PHY_Net, "PHY_dataset_random_0.8.npz", "logs")
