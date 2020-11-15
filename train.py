import torch
import torch.nn as nn
import os, glob
import optuna
import numpy as np
from CNN import CNN
from DataHelper import BertHelper
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup


gpu = torch.device('cuda:0')    


# compute precision@k function 
def precision_k(target_b, outputs_b, k=1):
    p_k_batch = np.empty(0, dtype=np.float32)
    for target, outputs in zip(target_b, outputs_b):
        target = np.where(target > 0)[0]
        outputs = np.argpartition(-outputs, k)[:k]
        accuracy = sum([(i in target) for i in outputs])
        p_k_value = [accuracy != 0.0 and accuracy / k or 0.0]
        p_k_batch = np.concatenate([p_k_batch, np.array(p_k_value)])
    return p_k_batch


class Model() :
    def __init__(self, params) :
        self.params = params
        torch.manual_seed(params['seed'])
        if params['cuda']:
            torch.cuda.manual_seed(params['seed'])
    

    # making dataloader, model and etc... 
    def build(self) :
        # Dataloaderの定義
        train_helper = BertHelper(self.params['train_path'])
        self.train_loader = DataLoader(
                                train_helper,        
                                batch_size = self.params['batch_size'],
                                shuffle = True
                            )
        val_helper = BertHelper(self.params['val_path'])
        self.val_loader = DataLoader(
                                val_helper,
                                batch_size = self.params['batch_size'], 
                                shuffle = True
                            )
        test_helper = BertHelper(self.params['test_path'])
        self.test_loader = DataLoader(
                                test_helper, 
                                batch_size = self.params['batch_size'], 
                                shuffle = True
                            )

        print('train sample:' + str(len(train_helper)) + ', val sample:' + str(len(val_helper)) + ', test sample:' + str(len(test_helper)))
        print('finished data regularization!')
        print('create model...', end = '')
        
        # CNNモデルの定義
        self.model = CNN(self.params)
        
        if self.params['cuda'] :
            self.model.cuda()

        # 損失関数の定義
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizerの定義，及び重み減衰を適応させるパラメータの洗濯
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params' : [p for i, p in self.model.named_parameters() if not any(j in i for j in no_decay)], 'weight_decay' : self.params['weight_decay']},
            {'params' : [p for i, p in self.model.named_parameters() if any(j in i for j in no_decay)], 'weight_decay' : 0.0},
            ]
        self.optimizer = AdamW(optimizer_parameters, lr = self.params['learning_rate'])

        # Schedulerの定義
        num_training_steps = len(self.train_loader) * self.params['epoch']
        num_warmup_steps = len(self.train_loader) * self.params['warmup_steps']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

        print('done!')
        print('training start')


    # Parameter search function
    def tuning(self, trial) :
        # CNNにおけるフィルタ数
        cnn_filter_num = trial.suggest_int('cnn_filter_num', 1, 4)
        
        # CNNにおけるフィルタのサイズ, filter_size ** filter_num
        cnn_filter_size = trial.suggest_int('cnn_filter_sizes', 2, 4)
        cnn_filter_sizes = [cnn_filter_size ** i for i in range(1, cnn_filter_num + 1)]
        
        # CNNにおけるConvolution層のストライド幅
        cnn_conv_stride = trial.suggest_int('cnn_conv_stride', 1, 8)
        
        # CNNにおけるMaxPooling層のストライド幅
        cnn_pool_stride = trial.suggest_int('cnn_pool_stride', 1, 8)
        
        # CNNにおけるDropout
        #cnn_dropout1 = trial.suggest_categorical('cnn_dropout1', [False, 0.25, 0.5, 0.75])
        cnn_dropout1 = False
        #cnn_dropout2 = trial.suggest_categorical('cnn_dropout2', [False, 0.25, 0.5, 0.75])
        cnn_dropout2 = False

        # CNNにおけるチャンネル数
        cnn_out_channels = trial.suggest_categorical('cnn_out_channels', [2 ** i for i in range(1, 8)])
        
        # CNNにおける全結合層の隠れ次元
        cnn_hidden_dim1 = trial.suggest_categorical('cnn_hidden_dim1', [2 ** i for i in range(5, 11)])
        
        # 学習率
        learning_rate = trial.suggest_loguniform('learning_rate', 0.0000001, 0.1)
        
        # WarmupSchedulerのWarmup地点
        #warmup_steps = trial.suggest_categorical('warmup_steps', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        warmup_steps = 0

        # Epoch
        #self.params['epoch'] = trial.suggest_categorical('epoch', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 閾値
        #threshold = trial.suggest_categorical('threshold', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        threshold = 0.5
        
        # 重み減衰
        weight_decay = trial.suggest_loguniform('weight_decay', 0.00001, 0.1)

        learning_params = {
            'batch_size' : 2,
            'learning_rate' : learning_rate,
            'warmup_steps' : warmup_steps,
            'threshold' : threshold,
            'weight_decay' : weight_decay
        }
        self.params.update(learning_params)

        cnn_params = {
            'cnn_out_channels' : cnn_out_channels,
            'cnn_filter_sizes' : cnn_filter_sizes,
            'cnn_hidden_dim1' : cnn_hidden_dim1,
            'cnn_conv_stride' : cnn_conv_stride,
            'cnn_pool_stride' : cnn_pool_stride,
            'cnn_dropout' : [cnn_dropout1, cnn_dropout2],
        }
        self.params.update(cnn_params)

        self.build()

        score = 0

        for epoch in range(1, self.params['epoch'] + 1) :
            self.train(epoch)    
            if epoch % self.params['epoch'] == 0 :
                score, val_loss, result = self.test('val')             

        return 1.0 - score
        

    # normal train&test function
    def run(self) :
        self.build()

        min_score = 0
        best_epoch = 0
        scores = []
        bad_counter = 0
        for epoch in range(1, self.params['epoch'] + 1) :
            self.train(epoch)
            
            # early stopping
            if epoch % 1 == 0 :
                score, val_loss, result = self.test('val')             
                scores.append(score)
                
            torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))

            if scores[-1] > min_score :
                min_score = scores[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.params['patience'] :
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            os.remove(file)

        torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
        
        print("Optimization Finished!")
        print('Loading {}th epoch'.format(best_epoch))
        self.model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        score, loss_mean, result = self.test('test')             

    
    def train(self, epoch) :
        with tqdm(self.train_loader) as bar :
            for i, (texts, labels) in enumerate(bar) :
                if self.params['cuda'] :
                    texts = texts.to(gpu)
                    labels = labels.to(gpu)
                
                self.model.train()
                self.optimizer.zero_grad()

                output = self.model(texts, labels)
                
                train_loss = self.criterion(output, labels)
                train_loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                
                bar.set_description('Train epoch %d' % epoch)
                bar.set_postfix(OrderedDict(loss = train_loss.item()))


    def test(self, mode) :
        self.model.eval()
        y_pre = np.empty((0, 103), dtype = np.float32)
        y_true = np.empty((0, 103), dtype = np.float32)

        p_1 = np.empty(0, dtype = np.float32)
        p_3 = np.empty(0, dtype = np.float32)
        p_5 = np.empty(0, dtype = np.float32)
        
        if mode == 'val' :
            loader = self.val_loader
        else :
            loader = self.test_loader

        with tqdm(loader) as bar :
            result = {}
            epoch_loss = [] 
            for i, (texts, labels) in enumerate(bar) :
                if self.params['cuda'] :    
                    texts = texts.to(gpu)
                    labels = labels.to(gpu)

                with torch.no_grad() :
                    output = self.model(texts)
                    test_loss = self.criterion(output, labels)
                    epoch_loss.append(test_loss.item())
                    output = torch.sigmoid(output)
                    output = output.cpu().detach_().numpy().copy()
                    labels = labels.cpu().detach_().numpy().copy()

                    output_threshold = (output > self.params['threshold'])
 
                    y_pre = np.append(y_pre, output_threshold, axis = 0)
                    y_true = np.append(y_true, labels, axis = 0)

                    p_1_batch = precision_k(labels, output, k=1)
                    p_3_batch = precision_k(labels, output, k=3)
                    p_5_batch = precision_k(labels, output, k=5)

                    bar.set_description(mode)
                    bar.set_postfix(OrderedDict(loss = test_loss.item()))

                p_1 = np.concatenate([p_1, p_1_batch])
                p_3 = np.concatenate([p_3, p_3_batch])
                p_5 = np.concatenate([p_5, p_5_batch])
        
        loss_mean = np.mean(epoch_loss)
        result['loss'] = loss_mean
        print('loss/val : ' + str(loss_mean))

        p_1 = np.mean(p_1)
        p_3 = np.mean(p_3)
        p_5 = np.mean(p_5)

        print('p@1 : ' + str(p_1))
        print('p@3 : ' + str(p_3))
        print('p@5 : ' + str(p_5))

        result['p@1'] = p_1
        result['p@3'] = p_3
        result['p@5'] = p_5

        macroF1 = f1_score(y_true, y_pre, average = 'macro', zero_division = 0)
        microF1 = f1_score(y_true, y_pre, average = 'micro', zero_division = 0)

        print('macro f1 : ' + str(macroF1))
        print('micro f1 : ' + str(microF1))

        result['macro'] = macroF1
        result['micro'] = microF1
        
        return macroF1, loss_mean, result