import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math

class Model(nn.Module):
	def __init__(self , bert_type = "bert-base-uncased" , relation_typs = 7):
		super().__init__()

		self.d_model = 768
		self.relation_typs = relation_typs
		self.no_rel = relation_typs - 1

		self.bert = BertModel.from_pretrained(bert_type).cuda()

		self.wq = nn.Linear(self.d_model , self.d_model)
		self.wk = nn.Linear(self.d_model , self.d_model)
		self.ln = nn.Linear(self.d_model , relation_typs)

	def forward(self , sents , ents):

		s = sents
		bs , n = sents.size()
		ne = max([len(x) for x in ents])
		d = self.d_model

		segm_index = s.new_zeros(s.size())
		posi_index = tc.arange(n).view(1,n).expand(bs , n).cuda()
		sent_mask  = (sents != 0)

		outputs  = self.bert(
			s , 
			token_type_ids = segm_index , 
			position_ids   = posi_index ,
			attention_mask = sent_mask , 
		) #(n , d)

		bert_encoded = outputs[0] #(bs , n , d)

		ent_mask = sent_mask.new_zeros( bs , ne ).float()
		ent_encode = bert_encoded.new_zeros( bs , ne , d )
		for _b in range(bs):
			for i , (u , v) in enumerate(ents[_b]):
				ent_encode[_b , i] = bert_encoded[_b , u : v , :].mean(dim = 0)
				ent_mask[_b , i] = 1

		q = self.wq(ent_encode)
		k = self.wk(ent_encode)

		alpha = q.view(bs,ne,1,d) * k.view(bs,1,ne,d) #(bs , n , n , d)
		alpha = self.ln(alpha)
		alpha = alpha * ent_mask.view(bs,ne,1,1) * ent_mask.view(bs,1,ne,1)

		return alpha


	def loss(self , pred , anss , ents):
		
		bs , ne , _ , d = pred.size()

		negative_rate = 0.5
		tot_pos_loss = 0
		tot_neg_loss = 0

		for _b in range(bs):

			pad_mask = tc.zeros(ne , ne).cuda().bool()
			pad_mask[:len(ents[_b]) , :len(ents[_b])] = 1

			rel_map = tc.zeros(ne , ne).cuda().long() + self.no_rel
			for u , v , t in anss[_b]:
				rel_map[u , v] = t

			pos_mask = (rel_map != self.no_rel)

			b_pred = -tc.log_softmax( pred[_b].view(-1,self.relation_typs) , dim = -1) #(ne*ne , rel_types)
			b_pred = b_pred[tc.arange(ne*ne) , rel_map.view(-1)].view(ne,ne)
			loss_map = b_pred

			if len(anss[_b]) > 0:
				pos_loss = loss_map.masked_select(pos_mask * pad_mask).mean()
			else:
				pos_loss = 0. #or it will be nan
			neg_loss = loss_map.masked_select((~pos_mask) * pad_mask).mean()

			try:
				assert not math.isnan(pos_loss)
				assert not math.isnan(neg_loss)
			except Exception:
				pdb.set_trace()

			tot_pos_loss = pos_loss if tot_pos_loss is None else tot_pos_loss + pos_loss
			tot_neg_loss = neg_loss if tot_neg_loss is None else tot_neg_loss + neg_loss

		tot_pos_loss /= bs
		tot_neg_loss /= bs

		return tot_pos_loss * (1-negative_rate) + tot_neg_loss * negative_rate

