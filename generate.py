import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

compare_idx = 0
topic_idx = 4

def generate_from_pred(pred , data_ent , relations , fil , no_rel , ans_rels = None):
	
	def add_rel(_b , i , j , t , fil):

		#只输出有relation的边的类型
		if ans_rels is not None:
			if (i,j) not in ans_rels[_b]:
				return

		reverse = False
		if i > j:
			i , j = j , i
			reverse = True
		t = relations[t]
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
				#pred[_b,i,j,topic_idx] *= 10 #more topic

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


class FakeFile:
	def __init__(self):
		self.buf = ""
	def write(self , x = ""):
		self.buf += x

def generate(preds , data_ent , relations , no_rel , ans_rels = None):
		
	#----- average predicted scores -----
	pred = 0
	for k in range(len(preds)):
		preds[k] = tc.softmax(preds[k] , dim = -1)
		pred += preds[k]
	pred /= len(preds)

	#----- generate from it -----
	fil = FakeFile()
	generate_from_pred(pred , data_ent , relations , fil , no_rel , ans_rels = ans_rels)

	return fil.buf