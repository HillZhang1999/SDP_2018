"""
根据CoNLL格式数据，生成语义依存图
"""

import logging
from typing import Dict, Tuple, List
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

logger = logging.getLogger(__name__)

class Relation(object):
    """
    关系类，定义和SDG中关系相关的数据结构。
    描述两个节点之间的关系，为某个实例化的节点对象（Node，主节点）添加从属的关系对象（Relation）。
    """
    type = None  # 定义主节点是另一个节点的孩子节点or头节点

    def __init__(self, node, rel):
        """
        初始化关系类
        :param node: 除主节点外，关系中的另一个节点
        :param rel: 关系类型
        """
        self.node = node
        self.rel = rel

    def __repr__(self):
        """
        展示当前关系类的内容
        :return: 当前关系类的内容
        """
        return "Node:{},Rel:{} || ".format(self.node, self.rel)


class Head(Relation):
    """
    继承关系类，定义关系中的另一个节点是主节点的头节点（Head）
    """
    type = 'HEAD'


class Child(Relation):
    """
    继承关系类，定义关系中的另一个节点是主节点的孩子节点（Child）
    """
    type = 'CHILD'


class Node(object):
    """
    节点类，定义和SDG中节点相关的数据结构。
    数据格式：CoNLL。
    """

    def __init__(self, index, label, pos_tag):
        """
        初始化节点类
        :param index:节点编号
        :param label:该节点的token
        :param pos_tag:节点词性
        """
        self.id = index  # 节点编号
        self.label = label  # 节点标签（指该节点的单词）
        self.pos_tag = pos_tag  # 节点词性
        self.heads, self.childs = [], []  # 头节点、孩子节点列表
        self.head_ids, self.child_ids = [], []  # 头节点、孩子节点索引列表

    def add_head(self, arc):
        """
        根据依存弧信息，为当前节点添加头节点
        :param arc: 依存弧信息
        """
        assert arc[0] == self.id
        self.heads.append(Head(arc[6], arc[7]))
        self.head_ids.append(arc[6])

    def add_child(self, arc):
        """
        根据依存弧信息，为当前节点添加孩子节点
        :param arc: 依存弧信息
        """
        assert arc[6] == self.id
        self.childs.append(Child(arc[0], arc[7]))
        self.child_ids.append(arc[0])


class Graph(object):
    """
    图类，定义和SDP中语义依存图相关的数据结构。
    根据输入的某个CoNLL格式的句子信息，创建该句对应的语义依存图。
    """

    def __init__(self, conll):
        """
        初始化图类
        :param conll: CoNLL格式的句子信息
        """
        self.meta_info = list(map(lambda x: x.split('\t'), conll.split('\n')))

        # 创建节点字典
        self.nodes = {0: Node(0, "ROOT", "ROOT")}
        for info in self.meta_info:
            info[0], info[6] = int(info[0]), int(info[6])
            if info[0] not in self.nodes.keys():
                self.nodes[info[0]] = Node(info[0], info[1], info[3])

        # 为节点添加依存关系
        for arc in self.meta_info:
            self.nodes[arc[6]].add_child(arc)
            self.nodes[arc[0]].add_head(arc)

    @property
    def arc_num(self):
        """
        :return:当前SDG中依存关系（边）数目
        """
        return len(self.meta_info)

    @property
    def node_num(self):
        """
        :return:当前SDG中节点数目（即句子中的token数目）
        """
        return len(self.nodes)

    def get_childs(self, id):
        """
        指定索引，获取该索引节点在SDG中的所有孩子节点
        :param id: 节点索引
        :return:孩子节点列表和孩子节点索引列表
        """
        childs = self.nodes[id].childs
        child_ids = [c.node for c in childs]
        return {"childs": childs,
                "child_ids": child_ids}

    def get_info(self):
        """
        获取当前句子的依存弧信息
        :return:一个字典，包含：token列表，词性列表，依存弧头尾节点索引列表，依存弧标签列表，CoNLL格式元数据，节点标签列表，依存弧列表, 字信息列表。
        """
        tokens, pos_tag, arc_indices, arc_tag, token_characters = [], [], [], [], []

        # 抽取信息
        for node_id in range(self.node_num):
            node = self.nodes[node_id]
            tokens.append(node.label)
            token_characters.append([ch for ch in node.label])
            pos_tag.append(node.pos_tag)
            for child in node.childs:
                arc_indices.append((child.node, node.id))  # tuple:(弧尾节点id,弧头节点id)
                arc_tag.append(child.rel)

        ret = {"tokens": tokens,
               "arc_indices": arc_indices,
               "arc_tag": arc_tag,
               "meta_info": self.meta_info,
               "pos_tag": pos_tag,
               "token_characters": token_characters
               }

        return ret


def parse_sentence(sentence_blob: str):
    """
    解析句子，生成SDG
    :param sentence_blob:CoNLL格式的句子信息
    :return:一个字典，包含：token列表，词性列表，依存弧头尾节点索引列表，依存弧标签列表，CoNLL格式元数据
    """
    graph = Graph(sentence_blob)
    ret = graph.get_info()
    return ret


def lazy_parse(text: str):
    """
    懒解析，生成一个generator，好处是节约内存。
    :param text:CoNLL格式文本
    :return:text中一个句子解析的结果（使用next()解析下一个句子）
    """
    conlls = text.split('\n\n')[:-1]
    for conll in conlls:
        yield parse_sentence(conll)


@DatasetReader.register("sdp_reader")  # 注册当前DatasetReader类，从而可用Json配置
class SDPDatasetReader(DatasetReader):
    """
    SDP的数据集读取类
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 action_indexers: Dict[str, TokenIndexer] = None,
                 arc_tag_indexers: Dict[str, TokenIndexer] = None,
                 characters_indexers: Dict[str, TokenIndexer] = None,
                 pos_tag_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        """
        初始化SDPDatasetReader
        :param token_indexers:token编号器
        :param action_indexers:转移动作编号器
        :param arc_tag_indexers:依存弧标签编号器
        :param characters_indexers:字标签编号器
        :param pos_tag_indexers:词性标签编号器
        :param lazy:是否懒加载
        """
        super().__init__(lazy)

        # TokenIndexer可以自动为token、action、arc-tag等进行编号
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._action_indexers = None
        if action_indexers is not None and len(action_indexers) > 0:
            self._action_indexers = action_indexers
        self._arc_tag_indexers = None
        if arc_tag_indexers is not None and len(arc_tag_indexers) > 0:
            self._arc_tag_indexers = arc_tag_indexers
        self._characters_tag_indexers = None
        if characters_indexers is not None and len(characters_indexers) > 0:
            self._characters_indexers = characters_indexers
        self._pos_tag_indexers = None
        if pos_tag_indexers is not None and len(pos_tag_indexers) > 0:
            self._pos_tag_indexers = pos_tag_indexers

    @overrides
    def _read(self, file_path: str):
        """
        重载read方法，这个函数从文本文件中获取样本数据，然后将样本数据转换成封装好的实例
        :param file_path:数据文件路径
        :return:一个样本的实例（instance），包含tokens, arc_tag, pos_tag, gold_actions, meta_info, token_characters六个域（field）
        """
        # 可以读取需要下载的网络地址
        file_path = cached_path(file_path)

        # 读取CoNLL格式的文件
        with open(file_path, 'r', encoding='utf-8') as fp:
            logger.info("Reading SDP instances from conll dataset at: %s", file_path)

            # 调用lazy_parse函数对文件内容进行解析。每次解析一个句子，并根据返回的字典生成一个实例。实例是每个样本多个域（或者称作字段，例如token域、词性域等等）的集合。
            for ret in lazy_parse(fp.read()):
                tokens = ret["tokens"]
                arc_indices = ret["arc_indices"]
                arc_tag = ret["arc_tag"]
                pos_tag = ret["pos_tag"]
                token_characters = ret["token_characters"]
                meta_info = ret["meta_info"]

                # CoNLL文件中不包含转移系统生成每个句子的语义依存图的正确转移序列，需要额外编写函数进行转移序列的解码（Root节点不需要传入）
                gold_actions = get_oracle_actions(tokens, arc_indices, arc_tag) if arc_indices else None

                if gold_actions and gold_actions[-1] == '-E-':
                    print('-E-')

                # 使用yield产生生成器
                yield self.text_to_instance(tokens, arc_indices, arc_tag, token_characters, gold_actions, meta_info, pos_tag)

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tag: List[str] = None,
                         token_characters: List[List[str]] = None,
                         gold_actions: List[str] = None,
                         meta_info: List = None,
                         pos_tag: List[str] = None) -> Instance:
        """
        文本转实例
        :param pos_tag: 词性列表
        :param tokens:token列表
        :param arc_indices:依存弧头尾节点索引列表
        :param arc_tag:依存弧标签列表
        :param token_characters:字信息列表
        :param gold_actions:生成SDG所需要的正确转移序列
        :param meta_info:CoNLL格式元数据
        :return:如果是训练阶段，则返回一个标注了SDG的句子样本的实例，包含五个域（field）；如果是预测阶段，则只返回一个tokens域的实例（Unlabelled data）
        """
        # 域字典
        fields: Dict[str, Field] = {}
        # token域
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        # 元数据字典
        meta_dict = {"tokens": tokens}

        # 依存弧头尾节点索引域
        if arc_indices is not None and arc_tag is not None:
            meta_dict["arc_indices"] = arc_indices
            meta_dict["arc_tag"] = arc_tag
            fields["arc_tag"] = TextField([Token(a) for a in arc_tag], self._arc_tag_indexers)

        # 正确转移序列域
        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        # 字信息序列域
        if token_characters is not None:
            meta_dict["token_characters"] = [TextField([Token(c) for c in chars], self._characters_indexers) for chars in token_characters]
            fields["token_characters"] = ListField([TextField([Token(c) for c in chars], self._characters_indexers) for chars in token_characters])

        # CoNLL格式元数据
        if meta_info is not None:
            # meta_dict["meta_info"] = meta_info[0]
            meta_dict["meta_info"] = meta_info

        # 词性标签域
        if pos_tag is not None:
            fields["pos_tag"] = TextField([Token(a) for a in pos_tag], self._pos_tag_indexers)
            meta_dict["pos_tag"] = pos_tag

        # 元数据域
        fields["metadata"] = MetadataField(meta_dict)

        # 将各域组合并实例化后，返回当前句子的实例
        return Instance(fields)


def get_oracle_actions(annotated_sentence, directed_arc_indices, arc_tag):
    """
    根据标注了SDG的句子，生成正确的转移序列。
    :param annotated_sentence:tokens列表
    :param directed_arc_indices:有向依存弧列表
    :param arc_tag:依存弧标签列表
    :return:转移动作序列
    """
    graph = {}
    for token_idx in range(len(annotated_sentence)):
        graph[token_idx] = []

    # 构建字典形式存储的语义依存图
    # 字典的键值对含义为：(孩子节点:[(头节点_1，弧标签_1),(头节点_2，弧标签_2)...])
    for arc, arc_tag in zip(directed_arc_indices, arc_tag):
        graph[arc[0]].append((arc[1], arc_tag))

    # N为节点个数，其中包含一个根节点ROOT
    N = len(graph)

    # 以列表形式存储的自顶向下的语义依存图，top_down_graph[i]为索引为i的节点所有孩子节点组成的列表
    top_down_graph = [[] for i in range(N)]  # N-1 real point, 1 root point => N point

    # sub_graph[i][j]表示索引为j的节点作为头节点，索引为i的节点作为孩子节点时，二者之间是否存在子图结构（连通）
    sub_graph = [[False for i in range(N)] for j in range(N)]

    # 生成top_down_graph
    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    actions = []  # 动作序列
    stack = [0]  # 正在处理的节点序列
    buffer = []  # 待处理的节点序列
    deque = []  # 暂时跳过的节点序列（可能存在多个头节点）

    # 待处理节点进入buffer
    for i in range(N - 1, 0, -1):
        buffer.append(i)

    def has_head(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1是否为w0的头节点
        """
        if w0 <= 0:
            return False
        for arc_tuple in graph[w0]:
            if arc_tuple[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的孩子节点
        """
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    def has_other_head(w):
        """
        :param w: 节点索引
        :return: w除了当前节点外是否还有其余头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    def lack_head(w):
        """
        :param w: 节点索引
        :return: w是否还有未找到的头节点
        """
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    def has_other_child_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余孩子节点
        """
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack and c != stack[-1] and not sub_graph[c][w]:
                return True
        return False

    def has_other_head_in_stack(w):
        """
        :param w: 节点索引
        :return: 除了栈顶节点外，w是否在栈中还有其余头节点
        """
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack and h[0] != stack[-1] and not sub_graph[w][h[0]]:
                return True
        return False

    def get_arc_label(w0, w1):
        """
        :param w0: 节点索引
        :param w1: 节点索引
        :return: w1作为头节点，w0作为孩子节点时，依存弧的标签
        """
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_actions_onestep():
        """
        根据当前stack、buffer、deque、actions四个部分，生成下一步的转移动作
        """
        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        # buffer首节点与栈顶节点有关系

        # 栈顶节点是buffer首节点的孩子节点，即生成弧的动作是"Left"
        if s0 > 0 and has_head(s0, b0):
            # 栈顶节点没有未找到的孩子节点或其余头节点，则直接将其出栈，执行"Left-Reduce"操作
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            # 否则需要将栈顶节点暂时入deque保存，以便之后重新进栈，执行"Left-Pass"操作。
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        # buffer首节点是栈顶节点的孩子节点，即生成弧的动作是"Right"
        elif s0 >= 0 and has_head(b0, s0):
            # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"Right-Shift"操作
            if not has_other_child_in_stack(b0) and not has_other_head_in_stack(b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                # Shift操作前，要将deque中暂存的节点先压栈
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            # buffer首节点在栈中除了栈顶节点以外，还有其他的孩子节点或者头节点，则将其暂时入deque保存，执行"Right-Pass"操作
            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        # buffer首节点与栈顶节点无关系，生成弧动作为"None"

        # buffer首节点在栈中除了栈顶节点以外，没有其他的孩子节点或者头节点，则将其进栈处理，执行"None-Shift"操作
        elif len(buffer) != 0 and not has_other_head_in_stack(b0) and not has_other_child_in_stack(b0):
            actions.append("NS")
            # Shift操作前，要将deque中暂存的节点先压栈
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        # 栈顶节点没有未找到的孩子节点或头节点，说明完成了所有依存关系的生成，可以出栈丢弃了，执行"None-Reduce"操作
        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        # 栈顶节点还有未找到的孩子节点或头节点，则将其暂时入deque保存，执行"None-Pass"操作
        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        # 如果出现了意料之外的分支，那么就说明出错了
        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    # 每次生成一步转移动作，终止条件为：buffer为空
    while len(buffer) != 0:
        get_oracle_actions_onestep()

    return actions
