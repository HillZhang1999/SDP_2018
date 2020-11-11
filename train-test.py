print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')
import os
import argparse
import moxing as mox
import torch
print(torch.__version__)

parser = argparse.ArgumentParser(description='Semantic Dependency Graph Parser')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')  
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')  
parser.add_argument('--train_dataset',
                    default='/home/work/modelarts/inputs/SDP_2018/data/SemEval-2016-master/train/text.train.conll',
                    help='Training dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。
# 验证集数据的路径
parser.add_argument('--validation_dataset',
                    default='/home/work/modelarts/inputs/SDP_2018/data/SemEval-2016-master/validation/text.valid.conll',
                    help='Validation dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。

args, unknown = parser.parse_known_args()

# /home/work/modelarts/inputs/SDP_2018/
init_path = os.path.abspath(os.path.join(os.path.dirname(args.train_dataset), '../../../'))
print(init_path)
if not os.path.exists(init_path):  # 检查OBS训练数据是否已拷贝到输入映射路径
    mox.file.copy_parallel(args.data_url, init_path)
    print('copy %s to %s' % (args.data_url, init_path))
else:
    print(init_path, 'already exists')
pwd = os.getcwd()
os.chdir(os.path.join(init_path, 'allennlp/'))
print(os.getcwd)
os.system('pip install --ignore-installed allennlp-1.2.0-py3-none-any.whl')
os.chdir(pwd)

print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')
print('loading allennlp')
import allennlp
print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')
print('loading success')