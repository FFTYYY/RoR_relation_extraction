import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

def generate_from_pred(pred , data_ent , relations , no_rel , gene_no_rel = False , ans_rels = None):

	bs , ne , _ , d = pred.size()
	gene_content = ["" for _ in range(bs)]

	def add_rel(_b , i , j , t):

		if t == no_rel:
			if (not gene_no_rel) or (i >= j):
				return
		if i == j:
			return #no self ring

		#只输出有relation的边的类型
		if ans_rels is not None:
			if (i,j) not in ans_rels[_b]:
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

		#----- small tricks to improve f1 value -----
		for i in range(len(data_ent[_b])):
			for j in range(len(data_ent[_b])):
				#pred[_b,i,j,topic_idx] *= 10 #more topic

				sym_relations = ["COMPARE" , "PER-SOC" , "PHYS"]
				for x in sym_relations:
					if x in relations:
						if i > j:
							pred[_b,i,j,relations.index(x) ] = 0 #no reverse compare
				#TODO
		#---------------------------------------------


		pred_map = pred[_b].max(-1)[1] #(ne , ne)

		try:
			assert (pred_map == pred_map).all()
		except AssertionError:
			pdb.set_trace()

		for i in range(len(data_ent[_b])):
			for j in range(i):
				add_rel(_b,i,j,int(pred_map[i , j]))
				add_rel(_b,j,i,int(pred_map[j , i]))

	return gene_content


def generate(preds , data_ent , relations , no_rel , gene_no_rel = False , ans_rels = None , 
		give_me_pred = False , split_generate = False):
		
	#----- average predicted scores -----
	pred = 0
	for k in range(len(preds)):
		preds[k] = tc.softmax(preds[k] , dim = -1)
		pred += preds[k]
	pred /= len(preds)

	#----- generate from it -----
	gene_cont = generate_from_pred(pred , data_ent , relations , no_rel , gene_no_rel , ans_rels = ans_rels)

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

	def get_no_rel_name(self):
		if self.gene_no_rel:
			return "gene_no_rel" # that means I don't want to tell you no_rel because it need to be generate
		return self.relations[self.no_rel]

	def __call__(self , preds , data_ent , ans_rels = None, give_me_pred = False, split_generate = False):
		return generate(preds , data_ent , self.relations , self.no_rel , self.gene_no_rel , 
				ans_rels , give_me_pred , split_generate)