import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math
from .loss_func import *
from .graph_encoder import Encoder

class Model(nn.Module):
	def __init__(self , bert_type = "bert-base-uncased" , relation_typs = 7 , dropout = 0.0):
		super().__init__()

		self.d_model = 768
		self.relation_typs = relation_typs
		self.no_rel = relation_typs - 1
		self.dropout = dropout


		self.bert = BertModel.from_pretrained(bert_type).cuda()
		self.bertdrop = nn.Dropout(self.dropout)

		self.wi = nn.Linear(self.d_model , self.d_model)
		self.drop = nn.Dropout(self.dropout)

		self.ent_emb = nn.Parameter(tc.zeros(2 , self.d_model))
		self.graph_enc = Encoder(h = 8 , d_model = self.d_model , hidden_size = 1024 , num_layers = 4)
		
		self.wu = nn.Linear(self.d_model , self.d_model)
		self.wv = nn.Linear(self.d_model , self.d_model)

		self.ln1 = nn.Linear(2 * self.d_model , self.d_model)
		self.wo = nn.Linear(self.d_model , relation_typs)

		self.reset_params()

	def reset_params(self):
		nn.init.xavier_normal_(self.wi.weight.data)
		nn.init.xavier_normal_(self.wu.weight.data)
		nn.init.xavier_normal_(self.wv.weight.data)
		nn.init.xavier_normal_(self.ln1.weight.data)
		nn.init.xavier_normal_(self.wo.weight.data)

		nn.init.constant_(self.wi.bias.data , 0)
		nn.init.constant_(self.wu.bias.data , 0)
		nn.init.constant_(self.wv.bias.data , 0)
		nn.init.constant_(self.ln1.bias.data , 0)
		nn.init.constant_(self.wo.bias.data , 0)


		nn.init.normal_(self.ent_emb.data , 0 , 0.01)

	def forward(self , sents , ents):

		bs , n = sents.size()
		ne = max([len(x) for x in ents])
		d = self.d_model
		s = sents

		ent_index = s.new_zeros(s.size())
		for _b in range(len(ents)):
			for u,v in ents[_b]:
				ent_index[_b , u:v] = 1

		#segm_index = s.new_zeros(s.size())
		posi_index = tc.arange(n).view(1,n).expand(bs , n).cuda()
		sent_mask  = (sents != 0)

		#with tc.no_grad():
		if True:
			outputs  = self.bert(
				s , 
				token_type_ids = ent_index , 
				position_ids   = posi_index ,
				attention_mask = sent_mask , 
			) #(n , d)

			bert_encoded = outputs[0] #(bs , n , d)
			bert_encoded = self.bertdrop(bert_encoded)

		ent_mask = sent_mask.new_zeros( bs , ne ).float()
		ent_encode = bert_encoded.new_zeros( bs , ne , d )
		for _b in range(bs):
			for i , (u , v) in enumerate(ents[_b]):
				ent_encode[_b , i] = bert_encoded[_b , u : v , :].mean(dim = 0)
				ent_mask[_b , i] = 1

		ee = self.wi(ent_encode)
		rel_enco = ee.view(bs,ne,1,d) + ee.view(bs,1,ne,d) #(bs , n , n , d)
		rel_enco = F.relu(rel_enco)
		#rel_enco[i,:,:,j] 是对称阵

		rel_enco = self.graph_encode(ent_encode , rel_enco , ents)

		u = self.wu(ent_encode)
		v = self.wv(ent_encode)
		alpha = u.view(bs,ne,1,d) + v.view(bs,1,ne,d) #(bs , n , n , d)
		alpha = F.relu(alpha)

		rel_enco = tc.cat([rel_enco , alpha] , dim = -1)
		#rel_enco = alpha
		rel_enco = F.relu(self.ln1(rel_enco))

		rel_enco = self.wo(rel_enco)

		return rel_enco

	def graph_encode(self , ent_encode , rel_encode , ents):

		bs , ne , d = ent_encode.size()

		ent_mask = tc.zeros(bs , ne , device = ent_encode.device)
		rel_mask = tc.zeros(bs , ne , ne , device = ent_encode.device)
		for _b in range(bs):
			ent_mask[_b , :len(ents[_b])] = 1
			rel_mask[_b , :len(ents[_b]) , :len(ents[_b])] = 1

		ent_encode = ent_encode + self.ent_emb[0].view(1,1,d)
		rel_encode = rel_encode + self.ent_emb[1].view(1,1,1,d)

		rel_encode , ent_encode = self.graph_enc(rel_encode , ent_encode , rel_mask , ent_mask)

		return rel_encode
