# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZERODNN(nn.Module):
    def __init__(self, config):
        super(ZERODNN, self).__init__()
        
        self.s_cnum = config['s_cnum']
        self.u_cnum = config['u_cnum']
        self.all_cnum = self.s_cnum + self.u_cnum 
        self.emb_len = config['emb_len']
        self.st_len = config['st_len']
        self.K = 300 # dimension of Convolutional Layer: lc
        self.L = 128 # dimension of semantic layer: y 
        
        self.batch_size = config['batch_size']
  
        self.linear = nn.Linear(self.K, self.L,bias = True) 
        self.mean = nn.AvgPool1d(self.st_len)
        self.max = nn.MaxPool1d(self.st_len)
        self.softmax = nn.Softmax()
        self.in_linear = nn.Linear(self.K, self.L,bias = False) 
        
        self.cossim = nn.CosineSimilarity(eps=1e-6)
        #self.criterion = torch.nn.MultiMarginLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.testmode = config['test_mode']
        self.dropout = nn.Dropout(0.9)
        
    def forward(self, utter, intents, embedding):
        
        
        if (embedding.nelement() != 0): 
            self.word_embedding = nn.Embedding.from_pretrained(embedding)
            
        utter = self.word_embedding(utter)      
        intents = self.word_embedding(intents)
        #utter = self.dropout(utter)
        utter = utter.transpose(1,2) 
        utter_mean = self.mean(utter) 
        utter_encoder = F.tanh(self.linear(utter_mean.permute(0,2,1))) 
        #utter_encoder = utter_encoder.squeeze(1)
        
        intents = intents.transpose(1,2) 
        class_num = list(intents.shape)
        
        int_encoders = [F.tanh(self.in_linear(intents[:,:,i])) for i in range(class_num[2])]
        int_encoders = torch.stack(int_encoders)
        '''
        sim = ([torch.bmm(utter_encoder,i.unsqueeze(2)) for i in int_encoders])
        sim = (torch.stack(sim)).transpose(0,1)
        sim = sim.view(class_num[0], class_num[2])
        '''
        sim = [self.cossim(utter_encoder.squeeze(1), yi) for yi in int_encoders]
        sim = torch.stack(sim)
        sim = sim.transpose(0,1)
        
        y_pred = [self.softmax(r) for r in sim]
        y_pred = torch.stack(y_pred)
        
        return y_pred
      # conv for intent and document at the same time
      
    def loss(self, y_pred, y_true): #y_red result y: target intent
        loss = self.criterion(y_pred, y_true)
        return loss