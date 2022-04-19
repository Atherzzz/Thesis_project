# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from Subsection_B_of_Section_VI_ViLiT.utils import pre_processing


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = 4
        self.hidden_size = 64
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.dropout_rate = 0.25

        self.query = Linear(self.hidden_size, self.hidden_size)
        self.key = Linear(self.hidden_size, self.hidden_size)
        self.value = Linear(self.hidden_size, self.hidden_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.proj_dropout = Dropout(self.dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # Narrow multi-head attention

        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        # attention(Q, K, V) = softmax(QK'/sqrt(dk))*V

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Need to use contiguous here to match size and stride
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_size = 64
        self.dropout_rate = 0.25
        self.fc1 = Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = Linear(self.hidden_size * 4, self.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(self.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 64
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = MLP()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attn(x)
        x = x + h
        x = self.attention_norm(x)

        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_block=3):
        super(Encoder, self).__init__()
        self.hidden_size = 64
        self.blocks = nn.ModuleList()
        self.encoder_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.num_block = num_block
        for _ in range(self.num_block):
            block = Block()
            self.blocks.append(copy.deepcopy(block))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.encoder_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_block=3):
        super(Transformer, self).__init__()
        self.input_size = 9
        self.hidden_size = 64
        self.num_patch = 300
        self.num_class = 10
        self.encoder = Encoder(num_block=num_block)
        self.fc1 = Linear(self.input_size, self.hidden_size)
        self.fc2 = Linear(self.hidden_size * self.num_patch, self.num_class)


    def forward(self, x):
        x = self.fc1(x)
        x = self.encoder(x)
        x = x.view([x.size()[0], x.size()[1] * x.size()[2]])
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    x = torch.randn((24, 60, 45))
    trans = Transformer()
    x = trans(x)
    print(x.size())
    trans = trans.to(device)

    print('test')
    X_train, y_train, X_val, y_val, X_test, y_test = pre_processing()
    X = np.array_split(X_train, 300)
    y = np.array_split(y_train, 300)

    for inputs, labels in zip(X, y):
        labels = [np.argwhere(e)[0][0] for e in labels]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.Tensor(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.view([inputs.size()[0], inputs.size()[1], inputs.size()[2] * inputs.size()[3]])
        print(inputs)
        y = trans(inputs)
        print(y)