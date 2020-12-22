import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel

# Convlution，MaxPooling層からの出力次元の算出用関数
def out_size(sequence_length, filter_size, padding = 0, dilation = 1, stride = 1):
    length = sequence_length + 2 * padding - dilation * (filter_size - 1) - 1
    length = int(length/stride)
    return length + 1


class CNN(torch.nn.Module):
    
    def __init__(self, params, gat = None):
        super(CNN, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        poolingLayer_out_size = 0
        
        self.dropout = params['cnn_dropout']
        self.filter_size = params['cnn_filter_sizes']
        
        if bool(self.dropout[0]) :
            self.drp1 = nn.Dropout(p = self.dropout[0])
        if bool(self.dropout[1]) :
            self.drp2 = nn.Dropout(p = self.dropout[1])

        for fsz in self.filter_size :        
            l_conv = nn.Conv1d(params['embedding_dim'], params['cnn_out_channels'], fsz, stride = params['cnn_conv_stride'])
            torch.nn.init.xavier_uniform_(l_conv.weight)

            l_pool = nn.MaxPool1d(params['cnn_pool_stride'], stride = params['cnn_pool_stride'])
            l_out_size = out_size(params['sequence_length'], fsz, stride = params['cnn_conv_stride'])
            pool_out_size = int(l_out_size * params['cnn_out_channels'] / params['cnn_pool_stride']) 
            poolingLayer_out_size += pool_out_size

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.linear1 = nn.Linear(poolingLayer_out_size, params['cnn_hidden_dim1'])
        self.linear2 = nn.Linear(params['cnn_hidden_dim1'], params['classes'])
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


    def forward(self, texts):

        texts = self.bert(texts)[0].detach_()

        texts = texts.permute(0, 2, 1)

        if bool(self.dropout[0]):
            texts = self.drp1(texts) 
        
        conv_out = []

        for i in range(len(self.filter_size)) :
            outputs = self.conv_layers[i](texts)
            outputs = outputs.view(outputs.shape[0], 1, outputs.shape[1] * outputs.shape[2])
            outputs = self.pool_layers[i](outputs)
            outputs = nn.functional.relu(outputs)
            outputs = outputs.view(outputs.shape[0], -1)
            conv_out.append(outputs)
            del outputs

        if len(self.filter_size) > 1 :
            outputs = torch.cat(conv_out, 1)
        else:
            outputs = conv_out[0]

        outputs = self.linear1(outputs)

        outputs = nn.functional.relu(outputs)

        if bool(self.dropout[1]) :
            outputs = self.drp2(outputs)
         
        outputs = self.linear2(outputs)

        return outputs
