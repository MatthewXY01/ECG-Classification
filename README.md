# ECG-Classification
Project of CS7304H: Theory and Methods for Statistical Learning [2020 Fall]

## Dataset

Download from https://www.kaggle.com/c/statlearning-sjtu-2020/data

## Run

for KNN:

`python run.py --method KNN --train_path ./data/ECGTrainData/Train --test_path ./data/ECGTestData/ECGTestData --dir_csv test.csv --n_neighbors 5`

for SVM:

`python run.py --method SVM --train_path ./data/ECGTrainData/Train --test_path ./data/ECGTestData/ECGTestData --dir_csv test.csv`

for DNN:

`python run.py --method MLP --train_path ./data/ECGTrainData/Train --test_path ./data/ECGTestData/ECGTestData --dir_csv test.csv`

You can see the validation accuracy on the command line and the prediction result for the testing set is in the 'test.csv'.

**Warning:** There may be some warnings when we apply wavelet decomposition to denoise the data, just ignore them! :) (Or you can refer to https://github.com/PyWavelets/pywt/issues/396)

## Requirement:

- PyTorch

- scikit-learn
- pywt
- py-ecg-detectors