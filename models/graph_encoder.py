import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb


class Attention(nn.Module):
	def __init__(self , d_model):
		self.d_model = d_model

		self.WQ = nn.Linear(d_model , d_model)
		self.WK = nn.Linear(d_model , d_model)
		self.WV = nn.Linear(d_model , d_model)

	def forward(self , R , E , R_mas , E_mas):
		'''
			R: (bs , ne , ne , d)
			E: (bs , ne , d)

			R_mas: (bs , ne , ne , 1)
			E_mas: (bs , ne , 1)
		'''

		bs , ne , d = E.size()
		assert d == self.d_model

		R_Q , R_K , R_V = self.WQ(R) , self.WK(R) , self.WV(R)
		E_Q , E_K , E_V = self.WQ(E) , self.WK(E) , self.WV(E)

		#from R to E
		alpha = (E_Q.view(bs,ne,1,d) * R_K).sum(-1) # (bs , ne , ne)
		alpha_mask = (E_mas.view(bs,ne,1) * R_mas.view(bs,ne,ne)).bool()
		alpha = alpha.masked_fill(~alpha_mask , float("-inf"))
		alpha = tc.softmax(alpha , dim = -1)
		E_Z = (alpha.view(bs,ne,ne,1) * R_V).sum(dim = 2)

		#from E to R
		beta_mask = E_mas.view(bs,ne,1) * E_mas.view(bs,1,ne) * R_mas.view(bs,ne,ne)
		beta_0 = R_Q * E_K.view(bs,ne,1,d).sum(-1)
		beta_1 = R_Q * E_K.view(bs,1,ne,d).sum(-1)

		beta_0 = beta_0.masked_fill(beta_mask , float("-inf"))
		beta_1 = beta_1.masked_fill(beta_mask , float("-inf"))

		new_beta_0 = 1 / (1 + tc.exp(beta_0 / beta_1))
		new_beta_1 = 1 / (1 + tc.exp(beta_1 / beta_0))
		beta_0 , beta_1 = new_beta_0 , new_beta_1

		R_Z = E_V.view(bs,ne,1,d) * beta_0.view(bs,ne,ne,1) + E.view(bs,1,ne,d) * beta_1.view(bs,ne,ne,1)

		R_Z *= R_mas
		E_Z *= E_mas

		return R_Z , E_Z

	
