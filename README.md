# SentimentAnalysis
A sentiment analysis of Amazon Reviews using Bidirectional LSTM. The network is trained with nearly 3,00,000 samples and gives more than 90% accuracy with less than 5 epochs of training. The network is carefully designed keeping the accuracy in mind.

1. To download the training dataset, kindly navigate to "Training_Data" file and click the link to download from Google Drive.
2. Change the number of epochs based on requirement. Also, the code is written to run only on GPU (as it uses CuDNNLSTM). If you want to train on CPU, then change CuDNNLSTM to "LSTM".
