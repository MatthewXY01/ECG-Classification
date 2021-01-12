import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='StatLearning SJTU 2020')
    parser.add_argument('--method', type=str, default='MLP')
    parser.add_argument('--train_path', type=str, default='./data/ECGTrainData/Train')
    parser.add_argument('--test_path', type=str, default='./data/ECGTestData/ECGTestData')
    parser.add_argument('--dir_csv', type=str, default='submission.csv')
    parser.add_argument('--n_neighbors', type=int, default=5)
    return parser