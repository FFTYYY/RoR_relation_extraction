import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

compare_idx = 0
topic_idx = 4

def generate_from_pred(relation_typs , no_rel , pred , data_ent , rel_id2name , fil):
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

	for _b in range(bs):

		#----- small tricks to improve f1 value -----
		for i in range(len(data_ent[_b])):
			for j in range(len(data_ent[_b])):
				pred[_b,i,j,topic_idx] *= 10 #more topic

				if i > j:
					pred[_b,i,j,compare_idx] = 0 #no reverse compare
		#---------------------------------------------


		pred_map = pred[_b].max(-1)[1] #(ne , ne)

		try:
			assert (pred_map == pred_map).all()
		except AssertionError:
			pdb.set_trace()

		for i in range(len(data_ent[_b])):
			for j in range(i):
				if pred_map[i , j] != no_rel:
					add_rel(_b,i,j,int(pred_map[i , j]),fil)
				if pred_map[j , i] != no_rel:
					add_rel(_b,j,i,int(pred_map[j , i]),fil)


def generate(relation_typs , no_rel , pred , data_ent , rel_id2name , fil):
		
		pred = tc.softmax(pred , dim = -1)

		generate_from_pred(relation_typs , no_rel , pred , data_ent , rel_id2name , fil)