"""
循环神经网络
"""
import collections
import math
import random
import re

import torch
from torch import nn

from base.util import plot, load_array, download_d2l_data, Timer, Accumulator, Animator, try_gpu
from learn.base_block import BaseNet


class SinPredict(BaseNet):

    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.learning_rate = 0.01
        self.epochs = 10

    def load_model(self):
        # 一个简单的多层感知机
        self.net = nn.Sequential(nn.Linear(4, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 1))
        # 初始化网络权重的函数
        self.net.apply(lambda m: nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear else None)
        # 平方损失。注意：MSELoss计算平方误差时不带系数1/2
        self.loss_func = nn.MSELoss(reduction='none')

    def train(self):
        trainer = torch.optim.Adam(self.net.parameters(), self.learning_rate)
        train_data_iter = self.get_train_data_iter()
        for epoch in range(self.epochs):
            for X, y in train_data_iter:
                trainer.zero_grad()
                loss = self.loss_func(self.net(X), y)
                loss.sum().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, loss: {self.evaluate_loss(train_data_iter):f}')


# 文本词表
class Vocab:

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = [] if tokens is None else tokens
        reserved_tokens = [] if reserved_tokens is None else reserved_tokens

        # 按出现频率排序
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    @staticmethod
    def count_corpus(tokens):
        """统计词元的频率"""
        # 这里的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    # 读取文章 time machine 内容
    @staticmethod
    def read_time_machine():
        """将时间机器数据集加载到文本行的列表中"""
        with open(download_d2l_data('timemachine.txt'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    @staticmethod
    def tokenize(lines, token='word'):
        """将文本行拆分为单词或字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            raise Exception('错误：未知词元类型：' + token)

    @staticmethod
    def load_corpus_time_machine(max_tokens=-1):  # @save
        """返回时光机器数据集的词元索引列表和词表"""
        lines = Vocab.read_time_machine()
        tokens = Vocab.tokenize(lines, 'char')
        vocab = Vocab(tokens)
        # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
        # 所以将所有文本行展平到一个列表中
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = self.seq_data_iter_random
        else:
            self.data_iter_fn = self.seq_data_iter_sequential
        self.corpus, self.vocab = Vocab.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    @staticmethod
    def seq_data_iter_random(corpus, batch_size, num_steps):
        """使用随机抽样生成一个小批量子序列"""
        # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
        corpus = corpus[random.randint(0, num_steps - 1):]
        # 减去1，是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中，
        # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        random.shuffle(initial_indices)

        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos: pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # 在这里，initial_indices包含子序列的随机起始索引
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)

    @staticmethod
    def seq_data_iter_sequential(corpus, batch_size, num_steps):
        """使用顺序分区生成一个小批量子序列"""
        # 从随机偏移量开始划分序列
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y

    @staticmethod
    def load_data_time_machine(batch_size, num_steps,
                               use_random_iter=False, max_tokens=10000):
        """返回时光机器数据集的迭代器和词表"""
        data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
        return data_iter, data_iter.vocab


class RNNModelScratch(BaseNet):

    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size

        self.normal_fn = lambda shape: torch.randn(size=shape, device=self.device) * 0.01
        self.three_normal_fn = lambda input_size: (self.normal_fn((input_size, self.hidden_size)),
                                                   self.normal_fn((self.hidden_size, self.hidden_size)),
                                                   torch.zeros(self.hidden_size, device=self.device))
        self.params = self.get_params()

    def __call__(self, x, state):
        x = nn.functional.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward(x, state)

    def parameters(self):
        return self.params

    def get_params(self):
        num_inputs = num_outputs = self.vocab_size

        # 隐藏层参数
        W_xh = self.normal_fn((num_inputs, self.hidden_size))
        W_hh = self.normal_fn((self.hidden_size, self.hidden_size))
        b_h = torch.zeros(self.hidden_size, device=self.device)
        # 输出层参数
        W_hq = self.normal_fn((self.hidden_size, num_outputs))
        b_q = torch.zeros(num_outputs, device=self.device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_state(self, batch_size):
        # 在初始化时返回隐状态，张量全⽤0填充，形状为（批量⼤⼩，隐藏单元数，隐状态可能包含多变量，返回元组
        return torch.zeros((batch_size, self.hidden_size), device=self.device),

    def begin_state(self, batch_size):
        return self.init_state(batch_size)

    def forward(self, inputs, state):
        """
        个时间步内计算隐状态和输出
        """
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = self.params
        H, = state
        outputs = []
        # X的形状：(批量大小，词表大小)
        # 循环神经⽹络模型通过inputs最外层的维度实现循环，以便逐时间步更新⼩批量数据的隐状态H
        for X in inputs:
            # 使⽤tanh函数作为激活函数，当元素在实数上满⾜均匀分布时，tanh函数的平均值为0
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)


class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)
        self.to(try_gpu())

    def forward(self, inputs, state):
        X = nn.functional.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device))


# ⻔控循环单元（gated recurrent units，GRU）
class GRU(RNNModelScratch):

    def get_params(self):
        num_inputs = num_outputs = self.vocab_size
        W_xz, W_hz, b_z = self.three_normal_fn(num_inputs)  # 更新门参数
        W_xr, W_hr, b_r = self.three_normal_fn(num_inputs)  # 重置门参数
        W_xh, W_hh, b_h = self.three_normal_fn(num_inputs)  # 候选隐状态参数
        # 输出层参数
        W_hq = self.normal_fn((self.hidden_size, num_outputs))
        b_q = torch.zeros(num_outputs, device=self.device)
        # 附加梯度
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device),

    def forward(self, inputs, state):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = self.params
        H, = state
        outputs = []
        for X in inputs:
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)


# ⻓短期记忆⽹络（long short-term memory，LSTM）
class LSTM(RNNModelScratch):

    def get_params(self):
        num_inputs = num_outputs = self.vocab_size
        W_xi, W_hi, b_i = self.three_normal_fn(num_inputs)  # 输入门参数
        W_xf, W_hf, b_f = self.three_normal_fn(num_inputs)  # 遗忘门参数
        W_xo, W_ho, b_o = self.three_normal_fn(num_inputs)  # 输出门参数
        W_xc, W_hc, b_c = self.three_normal_fn(num_inputs)  # 候选记忆元参数
        # 输出层参数
        W_hq = self.normal_fn((self.hidden_size, num_outputs))
        b_q = torch.zeros(num_outputs, device=self.device)
        # 附加梯度
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_state(self, batch_size):
        return (torch.zeros((batch_size, self.hidden_size), device=self.device),
                torch.zeros((batch_size, self.hidden_size), device=self.device))

    def forward(self, inputs, state):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = self.params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, C)


class RNNTrainer(object):

    def __init__(self, net: RNNModelScratch|nn.Module, loss_fn=nn.CrossEntropyLoss(), **kwargs):
        self.net: RNNModelScratch|nn.Module = net
        self.loss_fn = loss_fn
        self.optimizer = None
        self.device = try_gpu()

    def grad_clipping(self, theta):
        """裁剪梯度"""
        if isinstance(self.net, nn.Module):
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            params = self.net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def train_epoch(self, train_iter, use_random_iter):
        """训练网络一个迭代周期（定义见第8章）"""
        state, timer = None, Timer()
        metric = Accumulator(2)  # 训练损失之和,词元数量
        for X, Y in train_iter:
            if state is None or use_random_iter:
                # 在第一次迭代或使用随机抽样时初始化state
                state = self.net.begin_state(batch_size=X.shape[0])
            else:
                if isinstance(self.net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()
            y = Y.T.reshape(-1)
            X, y = X.to(self.device), y.to(self.device)
            y_hat, state = self.net(X, state)
            loss = self.loss_fn(y_hat, y.long()).mean()
            if isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer.zero_grad()
                loss.backward()
                self.grad_clipping(1)
                self.optimizer.step()
            else:
                loss.backward()
                self.grad_clipping(1)
                # 因为已经调用了mean函数
                self.optimizer(batch_size=1)
            metric.add(loss * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    def train(self, train_iter, vocab, learning_rate, epochs_size, use_random_iter=False):
        """训练模型（定义见第8章）"""
        loss = nn.CrossEntropyLoss()
        animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, epochs_size])
        # 初始化
        self.optimizer = torch.optim.SGD(self.net.parameters(), learning_rate)
        predict = lambda prefix: self.predict(prefix, 50, vocab)
        # 训练和预测
        for epoch in range(epochs_size):
            ppl, speed = self.train_epoch(train_iter, use_random_iter)
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(self.device)}')
        print(predict('time traveller'))
        print(predict('traveller'))

    def predict(self, prefix, predict_step_size, vocab):
        """在prefix后面生成新字符"""
        state = self.net.begin_state(batch_size=1)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=self.device).reshape((1, 1))
        for y in prefix[1:]:  # 预热期
            _, state = self.net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(predict_step_size):  # 预测 predict_step_size 步
            y, state = self.net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])


def test_sin_predict(show_plot=False):
    # 根据T生成sin函数值，并加入sigma=0.2的正态扰动
    T = 1000
    tau = 4
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    if show_plot:
        plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3)).show()
    # 将数据映射为数据对，x_t-1 -> x_t-tau
    # 将数据映射为数据对，x_t-1 -> x_t-tau
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))

    # 简单神经网络训练
    sin_predict = SinPredict()
    train_data_size = 600
    train_data_iter = load_array(features[:train_data_size], labels[:train_data_size], batch_size=16, is_train=True)
    sin_predict.set_data_iter(train_data_iter)
    sin_predict.train()

    # 预测 t+1 步
    onestep_predicts = sin_predict.net(features)
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    # plot([time, time[tau:]], [x.detach().numpy(), onestep_predicts.detach().numpy()], 'time',
    #      'x', legend=['data', '1-step predicts'], xlim=[1, 1000], figsize=(6, 3)).show()

    # 多步预测
    multistep_predicts = torch.zeros(T)
    multistep_predicts[: train_data_size + tau] = x[: train_data_size + tau]
    for i in range(train_data_size + tau, T):
        multistep_predicts[i] = sin_predict.net(multistep_predicts[i - tau:i].reshape((1, -1)))
    plot([time, time[tau:], time[train_data_size + tau:]], [x.detach().numpy(), onestep_predicts.detach().numpy(),
         multistep_predicts[train_data_size + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3)).show()


def test_vocab():
    lines = Vocab.read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])
    tokens = Vocab.tokenize(lines)
    for i in range(11):
        print(tokens[i])

    vocab = Vocab(tokens)
    # 将每一条文本行转换成一个数字索引列表
    print(list(vocab.token_to_idx.items())[:10])
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def test_rnn_scratch():
    batch_size, num_steps, hidden_size = 32, 35, 512
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    input_size = len(vocab)
    X = torch.arange(10).reshape((2, 5))
    rnn_net = RNNModelScratch(input_size, hidden_size)
    state = rnn_net.init_state(X.shape[0])
    Y, new_state = rnn_net(X.to(rnn_net.device), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    num_epochs, lr = 500, 1
    trainer = RNNTrainer(rnn_net)
    trainer.train(train_iter, vocab, lr, num_epochs)

    # 简洁实现的GRU
    gru_layer = nn.GRU(input_size, hidden_size)
    model = RNNModel(gru_layer, input_size)
    trainer = RNNTrainer(model)
    trainer.train(train_iter, vocab, lr, num_epochs)

    # 简洁实现的LSTM
    lstm_layer = nn.LSTM(input_size, hidden_size)
    model = RNNModel(lstm_layer, input_size)
    trainer = RNNTrainer(model)
    trainer.train(train_iter, vocab, lr, num_epochs)

    # 多层循环神经⽹络所
    vocab_size, hidden_size, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, hidden_size, num_layers)
    model = RNNModel(lstm_layer, len(vocab))
    trainer = RNNTrainer(model)
    trainer.train(train_iter, vocab, lr, num_epochs)

    # 双向循环神经⽹络使⽤了过去的和未来的数据，用得比较少
    lstm_layer = nn.LSTM(num_inputs, hidden_size, bidirectional=True)
    model = RNNModel(lstm_layer, len(vocab))
    trainer = RNNTrainer(model)
    trainer.train(train_iter, vocab, lr, num_epochs)


if __name__ == '__main__':
    # test_sin_predict()
    # test_vocab()
    test_rnn_scratch()
