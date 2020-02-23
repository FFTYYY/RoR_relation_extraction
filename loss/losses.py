import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel , BertTokenizer
import pdb
import math


def loss_1(pred , anss , ents , no_rel , class_weight , pad_ix = -100):
	'''
		直接平均，按类别加权
		and unweighted avg
	'''
	import numpy as np
	bs , ne , _ , d = pred.size()

	if no_rel < 0:
		no_rel = pad_ix #ignore index

	num = 0
	rel_map2 = np.zeros((bs, ne, ne))+no_rel
	_ = [[rel_map2.itemset((i,u,v),t) for u,v,t in b] for i,b in enumerate(anss)]
	rel_map2 = tc.LongTensor(rel_map2).to(pred.device)
	for _b in range(bs):
		tmp = rel_map2[_b] * 0 + pad_ix

		t_tmp = rel_map2[_b][:len(ents[_b]) , :len(ents[_b])]
		#t_tmp = tc.tril(rel_map2[_b][:len(ents[_b]) , :len(ents[_b])] , diagonal = -1)
		#t_tmp = t_tmp - tc.triu( tc.ones(t_tmp.size() , device = t_tmp.device) ) * 100

		tmp[:len(ents[_b]) , :len(ents[_b])] = t_tmp
		rel_map2[_b] = tmp
		num += len(ents[_b]) * len(ents[_b])

	#assert num == (rel_map2!=-100).long().sum()

	#----- style 1 -----
	loss_f = F.cross_entropy(
		pred.view(-1, pred.size(-1)), rel_map2.view(-1),
		weight=tc.FloatTensor(class_weight).to(pred), ignore_index=pad_ix , reduction = "sum")
	loss_f = loss_f / num

	assert float(loss_f) == float(loss_f)

	return loss_f

def loss_2(pred , anss , ents , no_rel , class_weight , pad_ix = -100):
	'''
		直接平均，按类别加权
		and weighted avg
	'''
	import numpy as np
	bs , ne , _ , d = pred.size()

	if no_rel < 0:
		no_rel = pad_ix #ignore index

	num = 0
	rel_map2 = np.zeros((bs, ne, ne))+no_rel
	_ = [[rel_map2.itemset((i,u,v),t) for u,v,t in b] for i,b in enumerate(anss)]
	rel_map2 = tc.LongTensor(rel_map2).to(pred.device)
	for _b in range(bs):
		tmp = rel_map2[_b] * 0 + pad_ix

		t_tmp = rel_map2[_b][:len(ents[_b]) , :len(ents[_b])]
		#t_tmp = tc.tril(rel_map2[_b][:len(ents[_b]) , :len(ents[_b])] , diagonal = -1)
		#t_tmp = t_tmp - tc.triu( tc.ones(t_tmp.size() , device = t_tmp.device) ) * 100

		tmp[:len(ents[_b]) , :len(ents[_b])] = t_tmp
		rel_map2[_b] = tmp
		num += len(ents[_b]) * len(ents[_b])

	loss_f = F.cross_entropy(
		pred.view(-1, pred.size(-1)), rel_map2.view(-1),
		weight=tc.FloatTensor(class_weight).to(pred), ignore_index=pad_ix)

	assert float(loss_f) == float(loss_f)

	return loss_f


def get_loss_func(name):
	return {
		"loss_1" : loss_1 , 
		"loss_2" : loss_2 , 
	}[name]
