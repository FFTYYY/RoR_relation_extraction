import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
from dataloader.data_config import data_config

def generate_from_pred(pred , data_ent , relations , no_rel , sym_rels = [] , gene_no_rel = False , 
			ans_rels = None):

	bs , ne , _ , d = pred.size()
	gene_content = ["" for _ in range(bs)]

	def add_rel(_b , i , j , t):

		if t == no_rel:
			if not gene_no_rel:
				return

		reverse = False
		if i > j:
			i , j = j , i
			reverse = True
		t = relations[t]

		gene_content[_b] += "%s(%s,%s%s)\n" % (
			t , 
			data_ent[_b][i].name , 
			data_ent[_b][j].name , 
			",REVERSE" if reverse else "" , 
		)


	for _b in range(bs):

		#----- for symmetric relations -----
		for i in range(len(data_ent[_b])):
			for j in range(len(data_ent[_b])):

				for x in sym_rels:
					if x in relations:
						if i > j:
							pred[_b,i,j,relations.index(x) ] = 0 # no reversed relation
		#---------------------------------------------


		pred_map = pred[_b].max(-1)[1] #(ne , ne)

		try:
			assert (pred_map == pred_map).all()
		except AssertionError:
			pdb.set_trace()

		if ans_rels is not None:
			for i,j in ans_rels[_b]:
				add_rel(_b,i,j,int(pred_map[i , j]))
		else:
			for i in range(len(data_ent[_b])):
				for j in range(i):
					add_rel(_b,i,j,int(pred_map[i , j]))
					add_rel(_b,j,i,int(pred_map[j , i]))

	return gene_content


def generate(preds , data_ent , relations , no_rel , sym_rels = [] , gene_no_rel = False , 
			ans_rels = None , give_me_pred = False , split_generate = False):
		
	#----- average predicted scores -----
	pred = 0
	for k in range(len(preds)):
		preds[k] = tc.softmax(preds[k] , dim = -1)
		pred += preds[k]
	pred /= len(preds)

	#----- generate from it -----
	gene_cont = generate_from_pred(pred , data_ent , relations , no_rel , sym_rels , 
			gene_no_rel , ans_rels = ans_rels)

	if not split_generate:
		gene_cont = "".join(gene_cont)

	if give_me_pred:
		return gene_cont , pred

	return gene_cont

class Generator:
	def __init__(self , C , relations , no_rel):
		self.gene_no_rel = C.gene_no_rel
		self.relations = relations
		self.no_rel = no_rel

		if C.dataset in data_config:
			self.sym_rels = data_config[C.dataset]["sym_relations"]
		else:
			self.sym_rels = []


	def get_no_rel_name(self): #for scorer
		return self.relations[self.no_rel]

	def __call__(self , preds , data_ent , ans_rels = None, give_me_pred = False, split_generate = False):
		return generate(preds , data_ent , self.relations , self.no_rel , self.sym_rels , 
			self.gene_no_rel , ans_rels , give_me_pred , split_generate)