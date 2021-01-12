import MLP
import KNN
import SVM
import argparse
from args import argument_parser

parser = argument_parser()
args = parser.parse_args()
if __name__ == "__main__":
    if args.method=='MLP':
        MLP.main(args.train_path, args.test_path, args.dir_csv)
    elif args.method =='SVM':
        SVM.main(args.train_path, args.test_path, args.dir_csv)
    elif args.method =='KNN':
        KNN.main(args.train_path, args.test_path, args.dir_csv, args.n_neighbors)
    else:
        print("There's no such a method called \'%s\'!"% (args.method))