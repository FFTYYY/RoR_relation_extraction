import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math
from .loss_func import *

class Model(nn.Module):
	def __init__(self , bert_type = "bert-base-uncased" , n_rel_typs = 7 , dropout = 0.0):
		super().__init__()

		self.d_model = 768
		self.n_rel_typs = n_rel_typs
		self.dropout = dropout

		self.bert = BertModel.from_pretrained(bert_type).cuda()
		self.bertdrop = nn.Dropout(self.dropout)

		self.wq = nn.Linear(self.d_model , self.d_model)
		self.wk = nn.Linear(self.d_model , self.d_model)
		self.drop = nn.Dropout(self.dropout)
		self.ln1 = nn.Linear(self.d_model , self.d_model)
		#self.ln2 = nn.Linear(2 * self.d_model , 2 * self.d_model)
		self.lno = nn.Linear(self.d_model , n_rel_typs)

		self.reset_params()

	def reset_params(self):
		nn.init.xavier_normal_(self.wq.weight.data)
		nn.init.xavier_normal_(self.wk.weight.data)
		nn.init.xavier_normal_(self.ln1.weight.data)
		#nn.init.xavier_normal_(self.ln2.weight.data)
		nn.init.xavier_normal_(self.lno.weight.data)

		nn.init.constant_(self.wq.bias.data , 0)
		nn.init.constant_(self.wk.bias.data , 0)
		nn.init.constant_(self.ln1.bias.data , 0)
		nn.init.constant_(self.lno.bias.data , 0)

	def forward(self , sents , ents):
		'''
		parameters:
			sents: LongTensor , (bsz , seq_len)
				其中0表示padding
			ents: list of list of tuples
				ents[b][i] = (u,v) ，表明第b个batch的第i个entity mention在sentence中的范围是[u,v)

		return: FloatTensor, (bsz , n_ent , n_ent , relation_typs)
			其中 relation_typs = self.relation_typs ，表示关系个数
			其中 ret[b,i,j,r]表示第b个batch的第i个entity和第j个entity之间的relation type是r的概率
			（没有softmax）
		'''

		s = sents
		bs , n = sents.size()
		ne = max([len(x) for x in ents])
		d = self.d_model

		ent_index = s.new_zeros(s.size())
		for _b in range(len(ents)):
			for u,v in ents[_b]:
				ent_index[_b , u:v] = 1


		segm_index = s.new_zeros(s.size())
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
		k = self.wk(ent_encode)
		alpha = q.view(bs,ne,1,d) + k.view(bs,1,ne,d) #(bs , n , n , d)
		alpha = F.relu(self.drop(alpha))
		alpha = F.relu(self.ln1(alpha))
		#alpha = F.relu(self.ln2(alpha))
		alpha = self.lno(alpha)


		alpha = alpha * ent_mask.view(bs,ne,1,1) * ent_mask.view(bs,1,ne,1)

		return alpha
		

