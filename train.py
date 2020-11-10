"""
用于本地训练的脚本
"""

import os
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
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '4'


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


def get_args():
    """
    从命令行读取参数
    """
    parser = argparse.ArgumentParser(description='Semantic Dependency Graph Parser')

    # 训练集数据的路径
    parser.add_argument('--train_dataset',
                        default='./SemEval-2016/train/text.train.conll',
                        help='Training dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。

    # 验证集数据的路径
    parser.add_argument('--validation_dataset',
                        default='./SemEval-2016/validation/text.valid.conll',
                        help='Validation dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。

    # 验证集数据的路径
    parser.add_argument('--test_dataset',
                        default='./SemEval-2016/test/text.test.conll',
                        help='Test dataset directory')  # 在ModelArts中创建算法时，必须进行输入路径映射配置，输入映射路径的前缀必须是/home/work/modelarts/inputs/，作用是在启动训练时，将OBS的数据拷贝到这个本地路径中供本地代码使用。

    # 数据载入器使用的子进程数目
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    # 梯度下降算法的学习率
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Learning rate')

    # 模型保存的路径
    parser.add_argument('--save_folder', default='./models',
                        help='Location to save checkpoint models')  # 在ModelArts中创建算法时，必须进行输出路径映射配置，输出映射路径的前缀必须是/home/work/modelarts/outputs/，作用是在训练结束时，将本地路径中的训练产生数据拷贝到OBS。

    # 用于训练的GPU数目
    parser.add_argument('--ngpu', default=1, type=int, help='gpu num for training')

    # batch规模
    parser.add_argument('--batch_size', default=30, type=int, help='batch size for training')

    # 训练轮数
    parser.add_argument('--epoch', default=30, type=int, help='total epochs for training')

    # 是否使用GPU训练
    parser.add_argument('--gpu_train', default=False, type=bool, help='whether use gpu for training')

    # 是否打乱训练集
    parser.add_argument('--shuffle', default=True, type=bool, help='whether shuffle the train-set')

    # 词向量维度
    parser.add_argument('--word_dim', default=100, type=int, help='word_embedding_dim')

    # 预训练词向量路径
    parser.add_argument('--word2vec_path', default="./data/giga.100.txt", type=str,
                        help='word2vec_pretrained_file_path')

    # 字向量维度
    parser.add_argument('--char_dim', default=100, type=int, help='char_embedding_dim')

    # 预训练词向量路径
    parser.add_argument('--char2vec_path', default="./data/giga.chars.100.txt", type=str,
                        help='char2vec_pretrained_file_path')

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

    # 模型保存路径
    parser.add_argument('--model_folder', default='./models/', type=str,
                        help='the path to save models')

    # 开发集预测结果保存路径
    parser.add_argument('--result_folder', default='./results/dev/', type=str,
                        help='the path to save results that our model predicted on validation set')

    # 验证集评价参数，用于early - stopping
    parser.add_argument('--validation_metric', default="+UF", type=str, help="""Validation metric to measure for whether to stop training using patience
        and whether to serialize an `is_best` model each epoch. The metric name
        must be prepended with either "+" or "-", which specifies whether the metric
        is an increasing or decreasing function""")

    args, unknown = parser.parse_known_args()  # 必须将parse_args改成parse_known_args，因为在ModelArts训练作业中运行时平台会传入一个额外的init_method的参数

    return args, unknown


def run_training_loop(args):
    """
    训练模型
    :param args: 命令行参数
    """
    manual_seed = 100
    torch.cuda.manual_seed(manual_seed)
    torch.manual_seed(manual_seed)

    # 读取语料数据
    reader = build_dataset_reader()
    train_set, dev_set = read_data(reader, args.train_dataset), read_data(reader, args.validation_dataset)
    vocab = build_vocab(train_set, word2vec_pretrained_file_path=args.word2vec_path,
                        char2vec_pretrained_file_path=args.char2vec_path)

    # 创建模型
    model = build_model(vocab=vocab,
                        reader=reader,
                        word_embedding_dim=args.word_dim,
                        word2vec_pretrained_file_path=args.word2vec_path,
                        char_embedding_dim=args.char_dim,
                        char2vec_pretrained_file_path=args.char2vec_path,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        pos_tag_embedding_dim=args.pos_tag_dim,
                        action_embedding_dim=args.action_dim,
                        model_save_folder=args.model_folder,
                        result_save_folder=args.result_folder
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
                               build_data_loaders(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
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


if __name__ == "__main__":
    # 读取参数
    args, unknown = get_args()
    run_training_loop(args)
