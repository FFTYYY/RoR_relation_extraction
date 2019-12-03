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

		self.ent_emb = nn.Embedding(num_embeddings = 2 , embedding_dim = self.d_model)

		self.bert = BertModel.from_pretrained(bert_type).cuda()
		self.bertdrop = nn.Dropout(self.dropout)

		self.wq = nn.Linear(self.d_model , self.d_model)
		#self.wk = nn.Linear(self.d_model , self.d_model)
		self.drop = nn.Dropout(self.dropout)
		self.lno = nn.Linear(self.d_model , relation_typs)

		self.graph_enc = Encoder(num_layers = 4 , d_model = self.d_model , d_hid = 1024 , h = 8 , drop_p = dropout)

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

		q = self.wq(ent_encode)
		#k = self.wk(ent_encode)
		alpha = q.view(bs,ne,1,d) + q.view(bs,1,ne,d) #(bs , n , n , d)
		#alpha[i,:,:,j] 是对称阵

		alpha = self.graph_encode(ent_encode , alpha , ents)
		alpha = self.lno(alpha)

		return alpha

	def graph_encode(self , ent_encode , rel_encode , ents):
		bs , ne , d = ent_encode.size()

		ent_masks = tc.zeros(bs , ne , device = ent_encode.device)
		rel_masks = tc.zeros(bs , ne , ne , device = ent_encode.device)
		for _b in range(bs):
			ent_masks[_b , :len(ents[_b])] = 1
			rel_masks[_b , :len(ents[_b]) , :len(ents[_b])] = 1

		return rel_encodes

	def generate(self , pred , data_ent , rel_id2name , fil):
		
		def add_rel(_b , i , j , t , fil):
			reverse = False
			if i > j:
				i , j = j , i
				reverse = True
			t = rel_id2name(t)
			fil.write("%s(%s,%s%s)\n" % (
				t , 
				data_ent[_b][i].name , 
				data_ent[_b][j].name , 
				",REVERSE" if reverse else "" , 
			))

		bs , ne , _ , d = pred.size()

		pred = tc.softmax(pred , dim = -1)

		for _b in range(bs):

			#----- a small trick to improve macro-f1 -----
			for i in range(len(data_ent[_b])):
				for j in range(len(data_ent[_b])):
					pred[_b,i,j,4] *= 10			
			#---------------------------------------------


			pred_map = pred[_b].max(-1)[1] #(ne , ne)

			#for i in range(len(data_ent[_b])):
			#	for j in range(len(data_ent[_b])):
			#		if pred[_b,i,j,pred_map[i,j]] < 0.2:
			#			pred_map[i,j] = self.no_rel

			try:
				assert (pred_map == pred_map).all()
			except AssertionError:
				pdb.set_trace()

			for i in range(len(data_ent[_b])):
				for j in range(i):
					if pred_map[i , j] != self.no_rel:
						add_rel(_b,i,j,int(pred_map[i , j]),fil)
					if pred_map[j , i] != self.no_rel:
						add_rel(_b,j,i,int(pred_map[j , i]),fil)

		

