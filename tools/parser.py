
import argparse

parser = argparse.ArgumentParser(description='--memory_length:可记忆的帧数\n'+\
                                 "--dataset_path:数据集根目录")

parser.add_argument('--epoch',type=int ,default=1000,help='')
parser.add_argument('--memory_length',type=int ,default=8,help='')
parser.add_argument('--dataset_path',type=str ,default="datasets/vid",help='')
parser.add_argument('--dictionary_path',type=str ,default="models/dictionary.gensim",help='')

args = parser.parse_args()
