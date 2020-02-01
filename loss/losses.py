import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math

def downsample(mask , count):
	if mask.long().sum() <= 0:
		return mask
	dropout_ratio = (1 - count.float() / mask.long().sum().float())

	mask = F.dropout(mask.float() , dropout_ratio) != 0

	return mask

def loss_4(pred , anss , ents , no_rel , class_weight = [1,0.5,0.5,1,5,0.5,1]):
	'''
		按类别加权，而且平衡正负例数量
	'''
	n_rel_typs = pred.size(-1)

	bs , ne , _ , d = pred.size()


	tot_loss = 0
	tot_show = 0

	pred = -tc.log_softmax( pred , dim = -1)
	device = pred.device
	for _b in range(bs):

		pad_mask = tc.zeros(ne , ne).to(device).bool()
		pad_mask[:len(ents[_b]) , :len(ents[_b])] = 1

		rel_map = tc.zeros(ne , ne).to(device).long() + no_rel
		for u , v , t in anss[_b]:
			rel_map[u , v] = t

		pos_count = 0

		loss_map = pred[_b].view(ne*ne,-1)[tc.arange(ne*ne) , rel_map.view(-1)].view(ne,ne)
		for c in range(n_rel_typs):

			c_mask = (rel_map == c) & pad_mask

			if c != no_rel:
				pos_count += c_mask.long().sum()
			else:
				c_mask = downsample(c_mask , pos_count)


			c_loss = loss_map.masked_select(c_mask)

			try:
				assert (c_loss == c_loss).all()
			except AssertionError:
				print ("bad loss")
				pdb.set_trace()

			tot_loss += c_loss.sum() * class_weight[c]
			tot_show += len(c_loss)

	tot_loss = tot_loss / tot_show

	return tot_loss



def loss_3(pred , anss , ents , no_rel , class_weight = [1,0.5,0.5,1,5,0.5,0.05] , pad_ix = -100):
	'''
		直接平均，按类别加权
	'''
	import numpy as np
	bs , ne , _ , d = pred.size()

	if no_rel < 0:
		no_rel = -100 #ignore index

	num = 0
	rel_map2 = np.zeros((bs, ne, ne))+no_rel
	_ = [[rel_map2.itemset((i,u,v),t) for u,v,t in b] for i,b in enumerate(anss)]
	rel_map2 = tc.LongTensor(rel_map2).to(pred.device)
	for _b in range(bs):
		tmp = rel_map2[_b] * 0 - 100
		tmp[:len(ents[_b]) , :len(ents[_b])] = rel_map2[_b][:len(ents[_b]) , :len(ents[_b])]
		rel_map2[_b] = tmp
		num += len(ents[_b]) * len(ents[_b])

	loss_f = F.cross_entropy(
		pred.view(-1, pred.size(-1)), rel_map2.view(-1),
		weight=tc.FloatTensor(class_weight).to(pred), ignore_index=-100, reduction='sum')
	loss_f = loss_f / num

	assert float(loss_f) == float(loss_f)

	return loss_f


def loss_1(pred , anss , ents , no_rel , class_weight = [1,0.5,0.5,1,5,0.5,1]):
	'''
		正负例分别加权平均
	'''
	n_rel_typs = pred.size(-1)


	bs , ne , _ , d = pred.size()

	neg_rate = 0.9
	tot_pos_loss = 0.
	tot_neg_loss = 0.

	pred = -tc.log_softmax( pred , dim = -1)
	device = pred.device
	for _b in range(bs):

		pad_mask = tc.zeros(ne , ne).to(device).bool()
		pad_mask[:len(ents[_b]) , :len(ents[_b])] = 1

		rel_map = tc.zeros(ne , ne).to(device).long() + no_rel
		for u , v , t in anss[_b]:
			rel_map[u , v] = t

		loss_map = pred[_b].view(ne*ne,-1)[tc.arange(ne*ne) , rel_map.view(-1)].view(ne,ne)

		pos_mask = (rel_map != no_rel)
		neg_mask = (rel_map == no_rel)


		pos_loss = loss_map.masked_select(pos_mask & pad_mask).mean() if len(anss[_b]) > 0 else 0.
		neg_loss = loss_map.masked_select(neg_mask & pad_mask).mean()

		tot_pos_loss += pos_loss
		tot_neg_loss += neg_loss

	tot_pos_loss /= bs
	tot_neg_loss /= bs
	tot_loss = tot_pos_loss * (1-neg_rate) + tot_neg_loss * neg_rate

	return tot_loss


def loss_2(pred , anss , ents , no_rel , class_weight = [1,0.5,0.5,1,5,0.5,1]):
	'''
		所有类分别平均然后加权平均
	'''
	n_rel_typs = pred.size(-1)

	bs , ne , _ , d = pred.size()

	neg_rate = 0.5

	tot_loss_class = [0.] * n_rel_typs
	tot_show_class = [0 ] * n_rel_typs

	pred = -tc.log_softmax( pred , dim = -1)
	device = pred.device
	for _b in range(bs):

		pad_mask = tc.zeros(ne , ne).to(device).bool()
		pad_mask[:len(ents[_b]) , :len(ents[_b])] = 1

		rel_map = tc.zeros(ne , ne).to(device).long() + no_rel
		for u , v , t in anss[_b]:
			rel_map[u , v] = t

		loss_map = pred[_b].view(ne*ne,-1)[tc.arange(ne*ne) , rel_map.view(-1)].view(ne,ne)
		for c in range(n_rel_typs):

			c_mask = (rel_map == c)
			c_loss = loss_map.masked_select(c_mask & pad_mask)

			try:
				assert (c_loss == c_loss).all()
			except AssertionError:
				pdb.set_trace()

			tot_loss_class[c] += c_loss.sum()
			tot_show_class[c] += len(c_loss)

	'''
	tot_loss = 0.
	for c in range(self.relation_typs):
		if tot_show_class[c] > 0:
			tot_loss = tot_loss + (tot_loss_class[c] / tot_show_class[c]) * class_weight[c]
	tot_loss /= sum(class_weight)
	'''
	pos_loss = sum(tot_loss_class[:-1]) / sum(tot_show_class[:-1])
	neg_loss = tot_loss_class[-1] / tot_show_class[-1]
	tot_loss = pos_loss*(1-neg_rate) + neg_loss*neg_rate

	return tot_loss

def get_loss_func(name):
	return {
		"loss_1" : loss_1 , 
		"loss_2" : loss_2 , 
		"loss_3" : loss_3 , 
		"loss_4" : loss_4 ,	
	}[name]
