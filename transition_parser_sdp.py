import logging
from typing import Dict, Optional, Any, List

import torch
from allennlp.data import Vocabulary, DatasetReader
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from torch.nn.modules import Dropout
from transition_sdp_predictor import sdp_trans_outputs_into_conll
from transition_sdp_metric import MyMetric
from stack_rnn import StackRnn
from supar_config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transition_parser_sdp2018")
class TransitionParser(Model):
    """
    解析器模型
    """

    def __init__(self,
                 vocab: Vocabulary,
                 reader: DatasetReader,
                 text_field_embedder: TextFieldEmbedder,
                 char_field_embedder: TextFieldEmbedder,
                 pos_tag_field_embedder: TextFieldEmbedder,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_layers: int,
                 metric: MyMetric = None,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 action_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 start_time: str = None,
                 model_save_folder: str = None,
                 result_save_folder: str = None
                 ) -> None:
        """
        模型构造类
        :param vocab:词典对象，包含padding后的所有tokens、pos_tag等内容及其索引
        :param text_field_embedder:文本域嵌入表示器
        :param char_field_embedder:字符域嵌入表示器
        :param pos_tag_field_embedder:词性域嵌入表示器
        :param word_dim:词向量嵌入维度
        :param hidden_dim:隐藏层维度
        :param action_dim:转移动作嵌入维度
        :param num_layers:隐藏层层数
        :param metric:评测器对象（计算F值等各类指标）
        :param recurrent_dropout_probability:递归神经网络递归时的dropout操作概率
        :param layer_dropout_probability:全连接隐藏层的dropout操作概率
        :param same_dropout_mask_per_instance:
        :param input_dropout:输入层的dropout操作概率
        :param action_embedding:动作序列嵌入器（封装了Pytorch的embedding类）
        :param initializer:参数初始化器
        :param regularizer:正则化器
        :param start_time:训练开始时间
        :param model_save_folder:保存checkpoint模型的文件夹路径
        :param result_save_folder:保存开发集预测结果文件的文件夹路径
        """
        super(TransitionParser, self).__init__(vocab, regularizer)

        self.epoch = 1
        self.best_UF = 0
        self.vocab = vocab
        self.reader = reader

        self.num_actions = vocab.get_vocab_size('actions')
        self.text_field_embedder = text_field_embedder
        self.token_characters_encoder = char_field_embedder
        self.pos_tag_field_embedder = pos_tag_field_embedder
        self.metric = metric
        self.start_time = start_time
        self.predicted_conlls = []
        self.model_save_path = model_save_folder
        self.result_save_path = result_save_folder

        self.args = Config().update(locals())

        # 动作序列嵌入器（封装了Pytorch的embedding类）
        self.action_embedding = action_embedding
        if action_embedding is None:
            self.action_embedding = Embedding(num_embeddings=self.num_actions,
                                              embedding_dim=action_dim,
                                              trainable=False)

        # 将buffer、stack、action_stack、deque四部分经过StackRNN编码得到的结果拼接，最后通过一个全连接层，得到维度为`hidden_dim`的向量
        self.p_s2h = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        # 将上一步得到的向量再通过一个全连接层，映射得到当前时刻各转移动作的得分（最后通过一个softmax激活函数转译为分值）
        self.p_act = torch.nn.Linear(hidden_dim, self.num_actions)

        # 初始状态下，buffer、stack、action_stack、deque四部分的嵌入结果（随机初始化，而不是直接使用零向量）
        self.pempty_buffer_emb = torch.nn.Parameter(torch.randn(hidden_dim))
        self.proot_stack_emb = torch.nn.Parameter(torch.randn(word_dim))
        self.pempty_action_emb = torch.nn.Parameter(torch.randn(hidden_dim))
        self.pempty_deque_emb = torch.nn.Parameter(torch.randn(hidden_dim))

        self._input_dropout = Dropout(input_dropout)

        # TODO stack可能需要采用TreeLSTM表示，buffer可能需要采用Bi-LSTM-Substraction表示
        # 四个部分的内容均采用StackRnn进行编码表示
        self.buffer = StackRnn(input_size=word_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               recurrent_dropout_probability=recurrent_dropout_probability,
                               layer_dropout_probability=layer_dropout_probability,
                               same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.stack = StackRnn(input_size=word_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.deque = StackRnn(input_size=word_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.action_stack = StackRnn(input_size=action_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=num_layers,
                                     recurrent_dropout_probability=recurrent_dropout_probability,
                                     layer_dropout_probability=layer_dropout_probability,
                                     same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        initializer(self)

    def _greedy_decode(self,
                       batch_size: int,
                       sent_len: List[int],
                       embedded_text_input: torch.Tensor,
                       oracle_actions: Optional[List[List[int]]] = None
                       ) -> Dict[str, Any]:
        """
        贪心解码，预测当前时刻的转移动作
        :param batch_size:batch数据规模
        :param sent_len:句子长度列表
        :param embedded_text_input:经过嵌入后的文本输入（对句子中的tokens进行嵌入表示）
        :param oracle_actions:正确转移动作序列
        :return:返回当预测结果和对应的loss
        """
        # TODO 将来是否要修改为Beam-Search解码？
        self.buffer.reset_stack(batch_size)
        self.stack.reset_stack(batch_size)
        self.deque.reset_stack(batch_size)
        self.action_stack.reset_stack(batch_size)

        losses = [[] for _ in range(batch_size)]
        edge_list = [[] for _ in range(batch_size)]

        # 依次将各句子的tokens嵌入结果轮流传入buffer编码器（由于这里的buffer编码器采用的是StackRnn，所以还需要倒序传入）
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    self.buffer.push(sent_idx,
                                     input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                                     extra={'token': sent_len[sent_idx] - token_idx})

        # 依次将个句子的stack初始嵌入向量（只存在Root时）传入stack编码器
        for sent_idx in range(batch_size):
            self.stack.push(sent_idx,
                            input=self.proot_stack_emb,
                            extra={'token': 0})

        # 创建action-id的映射。一个action可能对应多个id，因为还需要加上生成的弧标签，如："LP:Exp"。
        action_id = {
            action_: [self.vocab.get_token_index(a, namespace='actions') for a in
                      self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        # 开始预测：每个时刻计算选择各转移动作的概率，并选择概率最大的动作。
        trans_not_fin = True
        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):
                # 如果sentence对应的buffer为空，则说明它的转移动作序列已经预测完成。
                if self.buffer.get_len(sent_idx) != 0:
                    trans_not_fin = True
                    valid_actions = []
                    valid_action_tbl = {}

                    # 根据当前的stack、buffer情况，判断哪些转移动作是合法的。
                    if self.stack.get_len(sent_idx) > 1 and self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['LR']
                        valid_actions += action_id['LP']
                        valid_actions += action_id['RP']

                    if self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['NS']
                        valid_actions += action_id['RS']  # ROOT,NULL

                    if self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['NR']
                        valid_actions += action_id['NP']

                    log_probs = None
                    action = valid_actions[0]

                    # 当一个转移动作是当前唯一选项时，模型将不会对其建模。
                    # 将当前四部分的StackRnn表示进行拼接，通过一个前馈神经网络计算各转移动作的概率
                    if len(valid_actions) > 1:
                        stack_emb = self.stack.get_output(sent_idx)
                        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                            else self.buffer.get_output(sent_idx)

                        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                            else self.action_stack.get_output(sent_idx)

                        deque_emb = self.pempty_deque_emb if self.deque.get_len(sent_idx) == 0 \
                            else self.deque.get_output(sent_idx)

                        p_t = torch.cat([buffer_emb, stack_emb, action_emb, deque_emb])
                        h = torch.tanh(self.p_s2h(p_t))
                        logits = self.p_act(h)[torch.tensor(valid_actions, dtype=torch.long, device=h.device)]
                        valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                        # 通过Softmax激活函数，将得分转化为概率
                        log_probs = torch.log_softmax(logits, dim=0)

                        action_idx = torch.max(log_probs, 0)[1].item()
                        # 如果存在oracle_actions，那么每一步直接让对应的oracle_action进入action_stack编码（训练阶段，防止错误的传播）。
                        action = valid_actions[action_idx]

                    # 如果不存在oracle_actions，那么每一步直接让预测得到的最大概率动作进入action_stack编码（预测阶段）。
                    if oracle_actions is not None:
                        action = oracle_actions[sent_idx].pop(0)

                    self.action_stack.push(sent_idx,
                                           input=self.action_embedding(
                                               torch.tensor(action, device=embedded_text_input.device)),
                                           extra={
                                               'token': self.vocab.get_token_from_index(action, namespace='actions')})

                    # 计算每个句子各个时刻的预测损失。
                    if log_probs is not None:
                        losses[sent_idx].append(log_probs[valid_action_tbl[action]])

                    # 计算当前转移动作（除了NS、NR、NP，它们不产生弧）产生的依存关系：head-支配词；modifier-从属词。
                    if action in action_id["LR"] or action in action_id["LP"] or \
                            action in action_id["RS"] or action in action_id["RP"]:
                        if action in action_id["RS"] or action in action_id["RP"]:
                            head = self.stack.get_stack(sent_idx)[-1]
                            modifier = self.buffer.get_stack(sent_idx)[-1]
                        else:
                            head = self.buffer.get_stack(sent_idx)[-1]
                            modifier = self.stack.get_stack(sent_idx)[-1]

                        (head_rep, head_tok) = (head['stack_rnn_output'], head['token'])
                        (mod_rep, mod_tok) = (modifier['stack_rnn_output'], modifier['token'])

                        # 如果oracle_actions为空，那么需要记录下当前的预测结果（预测阶段）
                        if oracle_actions is None:
                            edge_list[sent_idx].append((mod_tok,
                                                        head_tok,
                                                        self.vocab.get_token_from_index(action, namespace='actions')
                                                        .split(':', maxsplit=1)[1]))

                    # 根据当前时刻的动作，更新四个部分的编码器内容。
                    # reduce
                    if action in action_id["LR"] or action in action_id["NR"]:
                        self.stack.pop(sent_idx)
                    # pass
                    elif action in action_id["LP"] or action in action_id["NP"] or action in action_id["RP"]:
                        stack_top = self.stack.pop(sent_idx)
                        self.deque.push(sent_idx,
                                        input=stack_top['stack_rnn_input'],
                                        extra={'token': stack_top['token']})
                    # shift
                    elif action in action_id["RS"] or action in action_id["NS"]:
                        while self.deque.get_len(sent_idx) > 0:
                            deque_top = self.deque.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=deque_top['stack_rnn_input'],
                                            extra={'token': deque_top['token']})

                        buffer_top = self.buffer.pop(sent_idx)
                        self.stack.push(sent_idx,
                                        input=buffer_top['stack_rnn_input'],
                                        extra={'token': buffer_top['token']})

        # 计算总的loss
        _loss = -torch.sum(
            torch.stack([torch.sum(torch.stack(cur_loss)) for cur_loss in losses if len(cur_loss) > 0])) / sum(
            [len(cur_loss) for cur_loss in losses])
        ret = {
            'loss': _loss,
            'losses': losses,
        }
        # 训练阶段不需要返回预测结果，每一步预测的结果只用于计算loss，随后即被丢弃。
        if oracle_actions is None:
            ret['edge_list'] = edge_list
        return ret

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                token_characters: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                arc_tag: torch.LongTensor = None,
                pos_tag: torch.LongTensor = None,
                ) -> Dict[str, torch.LongTensor]:
        """
        前向传播函数
        :param pos_tag: 词性列表
        :param token_characters: 每个单词划分为字后的列表
        :param arc_tag: 依存弧标签列表
        :param tokens: tokens列表
        :param metadata:元数据字典
        :param gold_actions:正确的转移序列
        :return:
        """
        batch_size = len(metadata)
        tokens['tokens']['tokens'] = tokens['tokens']['tokens'][:, 1:]
        pos_tag["pos_tag"]["tokens"] = pos_tag["pos_tag"]["tokens"][:, 1:]
        sent_len = [len(d['tokens']) - 1 for d in metadata]
        meta_info = [d['meta_info'] for d in metadata]

        token_characters["token_characters"]["token_characters"] = token_characters["token_characters"][
                                                                       "token_characters"][:, 1:].squeeze(3)

        oracle_actions = None
        token_characters = self.token_characters_encoder(token_characters)
        embedded_pos_tag = self.pos_tag_field_embedder(pos_tag)

        if gold_actions is not None:
            oracle_actions = [d['gold_actions'] for d in metadata]
            oracle_actions = [[self.vocab.get_token_index(s, namespace='actions') for s in l] for l in oracle_actions]

        # 句子中各tokens的嵌入表示：由词性向量（50维）、字向量（100维、giga语料）、词向量（100维、giga语料）得到
        embedded_text_input = self.text_field_embedder(tokens)
        embedded_text_input = torch.cat((embedded_text_input, token_characters, embedded_pos_tag), dim=-1)

        # 训练模式
        if self.training:
            embedded_text_input = self._input_dropout(embedded_text_input)
            ret_train = self._greedy_decode(batch_size=batch_size,
                                            sent_len=sent_len,
                                            embedded_text_input=embedded_text_input,
                                            oracle_actions=oracle_actions)

            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            return output_dict

        training_mode = self.training

        # 预测模式
        self.eval()
        with torch.no_grad():
            ret_eval = self._greedy_decode(batch_size=batch_size,
                                           sent_len=sent_len,
                                           embedded_text_input=embedded_text_input)

        # 在训练初期，弧表肯定是空的。
        # edge_list是一个列表，它含有batch_size个子列表。每个子列表放着表示一个样本SDG的元组。
        # 每个元组表示一条边，代表SDG中的一个语义依存关系，格式为：(尾节点，头节点，关系标签)。
        self.train(training_mode)
        edge_list = ret_eval['edge_list']
        _loss = ret_eval['loss']

        pos_tag = [d['pos_tag'] for d in metadata]
        output_dict = {
            'tokens': [d['tokens'] for d in metadata],
            'edge_list': edge_list,
            'meta_info': meta_info,
            'pos_tag': pos_tag,
            'loss': _loss
        }

        # 评价模型
        self.metric(edge_list, metadata, None)

        # 保存CoNLL格式化的结果
        for sent_idx in range(batch_size):
            if len(output_dict['edge_list'][sent_idx]) <= 5 * len(output_dict['tokens'][sent_idx]):
                self.predicted_conlls.append(sdp_trans_outputs_into_conll({
                    'tokens': output_dict['tokens'][sent_idx],
                    'edge_list': output_dict['edge_list'][sent_idx],
                    'pos_tag': output_dict['pos_tag'][sent_idx],
                }))

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        计算评价指标
        :param reset:是否重置评价参数
        :return:
        评价指标字典:
        1)LF:带标签的依存弧预测F1值；
        2)UF:不带标签的依存弧预测F1值；
        3)n_total:当前数据集中的依存弧总数；
        4)n_predict:预测得到的依存弧总数；
        """
        all_metrics: Dict[str, float] = {}
        if self.metric is not None and not self.training:
            all_metrics.update({'LF': self.metric.LF,
                                'UF': self.metric.UF,
                                'n_total': self.metric.n_total,
                                'n_predict': self.metric.n_predict,
                                # 'correct_arcs': self.metric.correct_arcs,
                                # 'correct_rels': self.metric.correct_rels,
                                # 'LF_Precision': self.metric.LF_Precision,
                                # 'UF_Precision': self.metric.UF_Precision,
                                # 'LF_Recall': self.metric.LF_Recall,
                                # 'UF_Recall': self.metric.UF_Recall
                                })
            if reset:
                if self.best_UF < self.metric.UF:
                    print('(best)saving model...')
                    self.best_UF = self.metric.UF
                    self.save(self.model_save_path + "model_{}.pt".format(self.start_time))
                    self.output_dev_results()
                self.epoch += 1
                print(f'\nepoch: {self.epoch}')
                self.metric.reset()
                self.predicted_conlls = []
        return all_metrics

    def output_dev_results(self):
        """
        输出开发集的预测结果
        """
        with open(self.result_save_path + "best_dev_results_{}.txt".format(self.start_time), "w", encoding="utf-8") as out:
            out.write(self.metric.__repr__())
            for r in self.predicted_conlls:
                if r:
                    for line in r:
                        out.write(line)
                    out.write("\n")
                    out.flush()

    @classmethod
    def load(cls, path, **kwargs):
        """
        载入模型
        :param path:模型保存路径
        :return: 模型对象
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(**state['args'])
        model.load_state_dict(state_dict=state['state_dict'], strict=False)
        model.to(device)

        return model

    def save(self, path):
        """
        保存模型
        :param path:模型保存路径
        """
        state_dict = self.state_dict()
        state = {
            'args': self.args,
            'state_dict': state_dict
        }
        torch.save(state, path)
