import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math
from .loss_func import *
from .graph_encoder import Encoder
from .matrix_transformer import Encoder as MatTransformer

class Model(nn.Module):
	def __init__(self , 
			bert_type = "bert-base-uncased" , n_rel_typs = 7 , dropout = 0.0 , 
			device = tc.device(0) ,
			gnn = False , matrix_trans = False , matrix_nlayer = 4 , 
		):
		super().__init__()

		self.gnn = gnn
		self.matrix_trans = matrix_trans

		self.d_model = 768
		self.n_rel_typs = n_rel_typs
		self.dropout = dropout
		self.device = device

		self.bert = BertModel.from_pretrained(bert_type).to(device)
		self.bertdrop = nn.Dropout(self.dropout)
		#self.pos_embedding = nn.Embedding(512 , self.d_model)

		if self.gnn:
			self.wi = nn.Linear(self.d_model , self.d_model)
		
		self.drop = nn.Dropout(self.dropout)

		if self.gnn:
			self.ent_emb = nn.Parameter(tc.zeros(2 , self.d_model))
			self.graph_enc = Encoder(h = 8 , d_model = self.d_model , hidden_size = 1024 , num_layers = 4)

		self.wu = nn.Linear(self.d_model , self.d_model)
		self.wv = nn.Linear(self.d_model , self.d_model)

		if self.gnn:
			self.ln1 = nn.Linear(2 * self.d_model , self.d_model)
		else:
			self.ln1 = nn.Linear(self.d_model , self.d_model)

		if self.matrix_trans:
			self.matt = MatTransformer(h = 8 , d_model = self.d_model , hidden_size = 4096 , num_layers = 4 , device = device)

		self.wo = nn.Linear(self.d_model , n_rel_typs)

		self.reset_params()

	def reset_params(self):

		if self.gnn:		
			nn.init.xavier_normal_(self.wi.weight.data)
		nn.init.xavier_normal_(self.wu.weight.data)
		nn.init.xavier_normal_(self.wv.weight.data)
		nn.init.xavier_normal_(self.ln1.weight.data)
		nn.init.xavier_normal_(self.wo.weight.data)

		if self.gnn:
			nn.init.constant_(self.wi.bias.data , 0)
		nn.init.constant_(self.wu.bias.data , 0)
		nn.init.constant_(self.wv.bias.data , 0)
		nn.init.constant_(self.ln1.bias.data , 0)
		nn.init.constant_(self.wo.bias.data , 0)

		if self.gnn:
			nn.init.normal_(self.ent_emb.data , 0 , 1e-2)
		#nn.init.normal_(self.pos_embedding.weight , 0 , 0.01)
		#nn.init.normal_(self.bert.embeddings.token_type_embeddings.weight , 0 , 0.01)

	def forward(self , sents , ents , devices = []):

		#----- head -----
		bs , n = sents.size()
		ne = max([len(x) for x in ents])
		d = self.d_model
		s = sents
		run_device = s.device.index

		if devices:
			device_number = devices.index(run_device)
		else:
			device_number = int(run_device)
		ents = ents[device_number * bs : (device_number + 1) * bs]

		#----- bert encoding -----

		ent_index = s.new_zeros(s.size())
		for _b in range(bs):
			for u,v in ents[_b]:
				ent_index[_b , u:v] = 1

		#segm_index = s.new_zeros(s.size())
		posi_index = tc.arange(n).view(1,n).expand(bs , n).to(s.device)
		sent_mask  = (sents != 0)

		outputs  = self.bert(
			s , 
			token_type_ids = ent_index , 
			position_ids   = posi_index ,
			attention_mask = sent_mask , 
		) #(n , d)

		bert_encoded = outputs[0] #(bs , n , d)
		#bert_encoded = bert_encoded + self.bert.embeddings.position_embeddings(posi_index)
		bert_encoded = self.bertdrop(bert_encoded)

		ent_encode = bert_encoded.new_zeros( bs , ne , d )
		for _b in range(bs):
			for i , (u , v) in enumerate(ents[_b]):
				ent_encode[_b , i] = bert_encoded[_b , u : v , :].mean(dim = 0)

		ent_mask , rel_mask = self.get_mask(ents , bs , ne , run_device)

		#----- gnn encoding -----
		if self.gnn:
			ee = self.wi(ent_encode)
			rel_enco = ee.view(bs,ne,1,d) + ee.view(bs,1,ne,d) #(bs , n , n , d)
			rel_enco = F.relu(rel_enco)
			# rel_enco[i,:,:,j] 是对称阵，为了让同一个节点收发的信息相同
			rel_enco = self.graph_encode(ent_encode , rel_enco , ent_mask , rel_mask)

		#----- naive encoding -----
		u = self.wu(ent_encode)
		v = self.wv(ent_encode)
		alpha = u.view(bs,ne,1,d) + v.view(bs,1,ne,d) #(bs , n , n , d)
		alpha = F.relu(alpha)

		#----- get final encode ----- 
		if self.gnn:
			rel_enco = tc.cat([rel_enco , alpha] , dim = -1)
			rel_enco = F.relu(self.ln1(rel_enco))
		else:
			rel_enco = F.relu(self.ln1(alpha))

		#----- matrix process -----
		if self.matrix_trans:
			rel_enco = self.matt(rel_enco , rel_mask)

		#----- ready to output -----
		rel_enco = self.wo(rel_enco)

		#alpha = alpha * ent_mask.view(bs,ne,1,1) * ent_mask.view(bs,1,ne,1)

		return rel_enco

	def get_mask(self , ents , bs , ne , run_device):
		ent_mask = tc.zeros(bs , ne , device = run_device)
		rel_mask = tc.zeros(bs , ne , ne , device = run_device)
		for _b in range(bs):
			ent_mask[_b , :len(ents[_b])] = 1
			rel_mask[_b , :len(ents[_b]) , :len(ents[_b])] = 1
		return ent_mask , rel_mask

	def graph_encode(self , ent_encode , rel_encode , ent_mask , rel_mask):

		bs , ne , d = ent_encode.size()

		ent_encode = ent_encode + self.ent_emb[0].view(1,1,d)
		rel_encode = rel_encode + self.ent_emb[1].view(1,1,1,d)

		rel_encode , ent_encode = self.graph_enc(rel_encode , ent_encode , rel_mask , ent_mask)

		return rel_encode
