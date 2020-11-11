import os
import argparse
import moxing as mox

def get_args():
    """
    从命令行读取参数
    """
    parser = argparse.ArgumentParser(description='Semantic Dependency Graph Parser')

    # 数据载入器使用的子进程数目
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    # 梯度下降算法的学习率
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Learning rate')

    # 用于训练的GPU数目
    parser.add_argument('--ngpu', default=1, type=int, help='gpu num for training')

    # batch规模
    parser.add_argument('--batch_size', default=30, type=int, help='batch size for training')

    # 训练轮数
    parser.add_argument('--epoch', default=30, type=int, help='total epochs for training')

    # 是否使用GPU训练
    parser.add_argument('--gpu_train', default=True, type=bool, help='whether use gpu for training')

    # 是否打乱训练集
    parser.add_argument('--shuffle', default=True, type=bool, help='whether shuffle the train-set')

    # 词向量维度
    parser.add_argument('--word_dim', default=100, type=int, help='word_embedding_dim')

    # 字向量维度
    parser.add_argument('--char_dim', default=100, type=int, help='char_embedding_dim')

    # 词性嵌入向量维度
    parser.add_argument('--pos_tag_dim', default=50, type=int, help='pos_tag_embedding_dim')

    # 动作嵌入向量维度
    parser.add_argument('--action_dim', default=50, type=int, help='action_embedding_dim')

    # MLP分类器隐藏层维度
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden_dim')

    # 隐藏层层数
    parser.add_argument('--num_layers', default=2, type=int, help='The number of hidden layers')

    # 模型在验证集上多少轮没有提升后训练结束
    parser.add_argument('--patience', default=5, type=int,
                        help='The training is stopped after `patience` epochs with no improvement')

    # 验证集评价参数，用于early - stopping
    parser.add_argument('--validation_metric', default="+UF", type=str, help="""Validation metric to measure for whether to stop training using patience
        and whether to serialize an `is_best` model each epoch. The metric name
        must be prepended with either "+" or "-", which specifies whether the metric
        is an increasing or decreasing function""")

    parser.add_argument('--seed', default=50, type=int,
                        help='the random seed')
    # 在ModelArts中创建训练作业时，必须指定OBS上的一个数据存储位置，启动训练时，会将该位置的数据拷贝到输入映射路径
    parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')  
    # 在ModelArts中创建训练作业时，必须指定OBS上的一个训练输出位置，训练结束时，会将输出映射路径拷贝到该位置
    parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')  
    # 训练集数据的路径
    parser.add_argument('--train_dataset',
                        default='/home/work/modelarts/inputs/SDP_2018/data/SemEval-2016-master/train/text.train.conll',
                        help='Training dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。
    # 验证集数据的路径
    parser.add_argument('--validation_dataset',
                        default='/home/work/modelarts/inputs/SDP_2018/data/SemEval-2016-master/validation/text.valid.conll',
                        help='Validation dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。
    # 验证集数据的路径
    parser.add_argument('--test_dataset',
                        default='/home/work/modelarts/inputs/SDP_2018/data/SemEval-2016-master/test/text.test.conll',
                        help='Test dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。
    # 预训练词向量路径
    parser.add_argument('--word2vec_path', default="/home/work/modelarts/inputs/SDP_2018/data/giga/giga.100.txt", type=str,
                        help='word2vec_pretrained_file_path')
    # 预训练字向量路径
    parser.add_argument('--char2vec_path', default="/home/work/modelarts/inputs/SDP_2018/data/giga/giga.chars.100.txt", type=str,
                        help='char2vec_pretrained_file_path')
    # 模型保存路径
    parser.add_argument('--model_folder', default='/home/work/modelarts/outputs/SDP_2018/models/', type=str,
                        help='the path to save models')    
    # 模型保存的路径
    parser.add_argument('--save_folder', default='/home/work/modelarts/outputs/SDP_2018/models/', type=str,
                        help='Location to save checkpoint models')  # 在ModelArts中创建算法时，必须进行输出路径映射配置，输出映射路径的前缀必须是/home/work/modelarts/outputs/，作用是在训练结束时，将本地路径中的训练产生数据拷贝到OBS。
    # 开发集预测结果保存路径
    parser.add_argument('--result_folder', default='/home/work/modelarts/outputs/SDP_2018/results/', type=str,
                        help='the path to save results that our model predicted on validation set')

    args, unknown = parser.parse_known_args()  # 必须将parse_args改成parse_known_args，因为在ModelArts训练作业中运行时平台会传入一个额外的init_method的参数

    return args, unknown

# 读取参数
args, unknown = get_args()
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


import time
from typing import Tuple
import torch
from allennlp.modules.token_embedders import TokenCharactersEncoder
from torch.utils.data import DataLoader
import allennlp
from allennlp.data import allennlp_collate
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from transition_sdp_reader import SDPDatasetReader
from transition_parser_sdp import TransitionParser
from transition_sdp_metric import MyMetric

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def build_trainer(
        model: Model,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        num_epochs: int,
        lr: float,
        validation_metric: str,
        patience: int
) -> Trainer:
    """
    创建模型训练器
    :param model: 模型对象
    :param train_loader: 训练集数据载入器
    :param dev_loader: 开发集数据载入器
    :param num_epochs: 训练轮数
    :param lr: 梯度下降算法的学习率
    :param patience: 多少轮没有提升后训练结束
    :param validation_metric: 验证集评价参数，用于early-stopping
    :return: 训练器
    """
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    # 使用Adam算法进行SGD
    optimizer = AdamOptimizer(parameters, lr=lr, betas=[0.9, 0.999])  # lr 1e-3
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        patience=patience,
        validation_metric=validation_metric
    )

    return trainer


def read_data(
        reader: DatasetReader,
        data_set_path: str,
) -> AllennlpDataset:
    """
    使用数据读取器，读取数据
    :param data_set_path: 数据集文件路径
    :param reader: 数据读取器
    :return: 数据集对象
    """
    return reader.read(data_set_path)


def build_dataset_reader() -> DatasetReader:
    """
    创建数据读取器
    :return: 数据读取器
    """
    token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)}
    characters_indexers = {'token_characters': TokenCharactersIndexer(namespace='token_characters')}
    action_indexers = {'actions': SingleIdTokenIndexer(namespace='actions')}
    arc_tag_indexers = {'arc_tag': SingleIdTokenIndexer(namespace='arc_tag')}
    pos_tag_indexers = {'pos_tag': SingleIdTokenIndexer(namespace='pos_tag')}

    return SDPDatasetReader(token_indexers=token_indexers,
                            action_indexers=action_indexers,
                            arc_tag_indexers=arc_tag_indexers,
                            characters_indexers=characters_indexers,
                            pos_tag_indexers=pos_tag_indexers
                            )


def build_vocab(train_instances: AllennlpDataset,
                word2vec_pretrained_file_path: str,
                char2vec_pretrained_file_path,
                ) -> Vocabulary:
    """
    创建词表
    :param char2vec_pretrained_file_path:预训练字向量路径
    :param word2vec_pretrained_file_path:预训练词向量路径
    :param train_instances:训练集
    :return:词典对象
    """
    instances = train_instances
    vocab = Vocabulary.from_instances(instances,
                                      pretrained_files={'tokens': word2vec_pretrained_file_path,
                                                        'token_characters': char2vec_pretrained_file_path},
                                      min_count={'tokens': 2, 'token_characters': 3}
                                      )
    return vocab


def build_model(vocab: Vocabulary,
                reader: DatasetReader,
                word_embedding_dim: int,
                word2vec_pretrained_file_path: str,
                char_embedding_dim: int,
                char2vec_pretrained_file_path: str,
                pos_tag_embedding_dim: int,
                action_embedding_dim: int,
                hidden_dim: int,
                num_layers: int,
                model_save_folder: str = None,
                result_save_folder: str = None
                ) -> Model:
    """
    创建模型
    :param vocab: 词表对象
    :param reader: 数据读取器
    :param word_embedding_dim: 词向量维度
    :param word2vec_pretrained_file_path: 预训练词向量路径
    :param char_embedding_dim: 字向量维度
    :param char2vec_pretrained_file_path: 预训练字向量路径
    :param pos_tag_embedding_dim: 词性嵌入向量维度
    :param action_embedding_dim: 动作嵌入向量维度
    :param hidden_dim: 隐藏层维度（用于分类）
    :param num_layers: 隐藏层层数
    :param model_save_folder:保存checkpoint模型的文件夹路径
    :param result_save_folder:保存开发集预测结果文件的文件夹路径
    :return: 模型对象
    """
    text_field_embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=word_embedding_dim,
                             pretrained_file=word2vec_pretrained_file_path,
                             vocab=vocab)})
    char_field_embedder = BasicTextFieldEmbedder(
        {"token_characters": TokenCharactersEncoder(Embedding(embedding_dim=char_embedding_dim,
                                                              pretrained_file=char2vec_pretrained_file_path,
                                                              vocab=vocab),
                                                    BagOfEmbeddingsEncoder(char_embedding_dim, True), 0.33)})
    pos_field_embedder = BasicTextFieldEmbedder(
        {"pos_tag": Embedding(embedding_dim=pos_tag_embedding_dim, num_embeddings=vocab.get_vocab_size("pos_tag"))})

    metric = MyMetric()
    action_embedding = Embedding(vocab_namespace='actions', embedding_dim=action_embedding_dim,
                                 num_embeddings=vocab.get_vocab_size('actions'))

    return TransitionParser(vocab=vocab,
                            reader=reader,
                            text_field_embedder=text_field_embedder,
                            char_field_embedder=char_field_embedder,
                            pos_tag_field_embedder=pos_field_embedder,
                            word_dim=word_embedding_dim + char_embedding_dim + pos_tag_embedding_dim,
                            hidden_dim=hidden_dim,
                            action_dim=action_embedding_dim,
                            num_layers=num_layers,
                            metric=metric,
                            recurrent_dropout_probability=0.2,
                            layer_dropout_probability=0.2,
                            same_dropout_mask_per_instance=True,
                            input_dropout=0.2,
                            action_embedding=action_embedding,
                            start_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                            model_save_folder=model_save_folder,
                            result_save_folder=result_save_folder
                            )


def build_data_loaders(
        data_set: AllennlpDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader, allennlp.data.DataLoader]:
    """
    创建数据载入器
    :param data_set: 数据集对象
    :param batch_size: batch大小
    :param num_workers: 同时使用多少个线程载入数据
    :param shuffle: 是否打乱训练集
    :return: 训练集、开发集、测试集数据载入器
    """
    return DataLoader(data_set, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                      collate_fn=allennlp_collate)

def run_training_loop(args):
    """
    训练模型
    :param args: 命令行参数
    """
    manual_seed = args.seed
    torch.cuda.manual_seed(manual_seed)
    torch.manual_seed(manual_seed)

    # 读取语料数据
    reader = build_dataset_reader()
    train_set, dev_set = read_data(reader, config['train_file_path']), read_data(reader, config['dev_file_path'])
    vocab = build_vocab(train_set, word2vec_pretrained_file_path=config['pretrain_path'],
                        char2vec_pretrained_file_path=config['pretrain_char_path'])

    # 创建模型
    model = build_model(vocab=vocab,
                        reader=reader,
                        word_embedding_dim=args.word_dim,
                        word2vec_pretrained_file_path=config['pretrain_path'],
                        char_embedding_dim=args.char_dim,
                        char2vec_pretrained_file_path=config['pretrain_char_path'],
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        pos_tag_embedding_dim=args.pos_tag_dim,
                        action_embedding_dim=args.action_dim,
                        model_save_folder=config['model_path'],
                        result_save_folder=config['dev_output_path']
                        )
    if args.gpu_train:
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # 为数据集的各域创建索引
    train_set.index_with(vocab)
    dev_set.index_with(vocab)

    # 创建数据载入器
    train_loader, dev_loader = build_data_loaders(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=args.shuffle), \
                               build_data_loaders(dev_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=args.shuffle)

    # 创建训练器
    trainer = build_trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=args.epoch,
        lr=args.lr,
        validation_metric=args.validation_metric,
        patience=args.patience
    )

    # 开始训练
    trainer.train()

config = {
    'pretrain_path': 'giga/giga.100.txt',
    'pretrain_char_path': 'giga/giga.chars.100.txt',
    'train_file_path' : 'SemEval-2016/train/text.train.conll',
    'dev_file_path' : 'SemEval-2016/validation/text.valid.conll',
    'model_path' : 'models/model_{}.pt',
    'dev_output_path' : "results/best_dev_results_{}.txt",
}

def adjust_config(config, args):
    # /home/work/modelarts/inputs/SDP_2018/
    train_input_dir = os.path.abspath(os.path.join(os.path.dirname(args.train_dataset), '../../../'))
    # dataset_dir = os.path.join(train_input_dir, 'data')
    print(train_input_dir)
    print(os.listdir(train_input_dir))
    # print(dataset_dir)
    # print(os.listdir(dataset_dir))
    
    config['train_file_path'] = os.path.join(train_input_dir, config['train_file_path'])
    config['dev_file_path'] = os.path.join(train_input_dir, config['dev_file_path'])
    config['pretrain_path'] = os.path.join(train_input_dir, config['pretrain_path'])
    config['pretrain_char_path'] = os.path.join(train_input_dir, config['pretrain_char_path'])


    # /home/work/modelarts/outputs/SDP_2018/
    train_output_dir = os.path.abspath(os.path.join(args.save_folder, '../'))
    print(train_output_dir)
    t_path = os.path.abspath(os.path.join(train_output_dir, '../../../'))
    print(t_path, 'is exist:', os.path.exists(t_path))
    if not os.path.exists(t_path):
        print('create dir', t_path)
        os.mkdir(t_path)
    t_path = os.path.abspath(os.path.join(train_output_dir, '../../'))
    print(t_path, 'is exist:', os.path.exists(t_path))
    if not os.path.exists(t_path):
        print('create dir', t_path)
        os.mkdir(t_path)
    t_path = os.path.abspath(os.path.join(train_output_dir, '../'))
    print(t_path, 'is exist:', os.path.exists(t_path))
    if not os.path.exists(t_path):
        print('create dir', t_path)
        os.mkdir(t_path)
    print(train_output_dir, 'is exist:', os.path.exists(train_output_dir))
    if not os.path.exists(train_output_dir):  # 检查OBS训练数据是否已拷贝到输入映射路径
        print('create dir', train_output_dir)
        os.mkdir(train_output_dir)
        os.mkdir(os.path.join(train_output_dir, 'models'))
        os.mkdir(os.path.join(train_output_dir, 'results'))
    else:
        print(train_output_dir, 'already exists')
    config['model_path'] = os.path.join(train_output_dir, config['model_path'])
    config['dev_output_path'] = os.path.join(train_output_dir, config['dev_output_path'])
    print(config['model_path'])
    print(config['dev_output_path'])

    return train_output_dir

if __name__ == "__main__":
    train_output_dir = adjust_config(config, args)
    run_training_loop(args)
    # 将训练输出拷贝到OBS
    if args.train_url.startswith('obs') or args.train_url.startswith('s3'):
        mox.file.copy_parallel(train_output_dir, args.train_url)
