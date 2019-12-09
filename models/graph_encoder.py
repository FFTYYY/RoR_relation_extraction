import sys
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb


class Attention(nn.Module):
	def __init__(self , h , d_model):
		super().__init__()

		assert d_model % h == 0

		self.d_model = d_model
		self.h = h
		self.dk = d_model // h

		self.WQ = nn.Linear(self.dk , self.dk)
		self.WK = nn.Linear(self.dk , self.dk)
		self.WV = nn.Linear(self.dk , self.dk)		

		#self.reset_params()

	def reset_params(self):
		nn.init.xavier_normal_(self.WQ.weight.data)
		nn.init.xavier_normal_(self.WK.weight.data)
		nn.init.xavier_normal_(self.WV.weight.data)

		nn.init.constant_(self.WQ.bias.data , 0)
		nn.init.constant_(self.WK.bias.data , 0)
		nn.init.constant_(self.WV.bias.data , 0)

	def forward(self , R , E , R_mas , E_mas):
		'''
			R: (bs , ne , ne , d)
			E: (bs , ne , d)

			R_mas: (bs , ne , ne , 1)
			E_mas: (bs , ne , 1)
		'''

		h , dk = self.h , self.dk
		bs , ne , d = E.size()
		assert d == self.d_model

		R = R.view(bs,ne,ne,h,dk).permute(0,3,1,2,4).contiguous() #(bs , h , ne , ne , dk)
		E = E.view(bs,ne,h,dk).permute(0,2,1,3).contiguous()      #(bs , h , ne , dk)
		R_mas = R_mas.view(bs,1,ne,ne,1)
		E_mas = E_mas.view(bs,1,ne,1)

		R_Q , R_K , R_V = self.WQ(R) , self.WK(R) , self.WV(R)
		E_Q , E_K , E_V = self.WQ(E) , self.WK(E) , self.WV(E)

		#from R to E
		alpha = (E_Q.view(bs,h,ne,1,dk) * R_K).sum(-1) # (bs , h , ne , ne)
		#alpha_mask = (E_mas.view(bs,ne,1) * R_mas.view(bs,ne,ne)).bool()
		alpha_mask = (E_mas.view(bs,1,1,ne).expand(bs,h,ne,ne)).bool() #防止出现一行全是-inf
		alpha = alpha.masked_fill(~alpha_mask , float("-inf"))
		alpha = tc.softmax(alpha , dim = -1)
		E_Z = (alpha.view(bs,h,ne,ne,1) * R_V).sum(dim = 2)

		#from E to R
		beta_0 = (R_Q * E_K.view(bs,h,ne,1,dk)).sum(-1 , keepdim = True)
		beta_1 = (R_Q * E_K.view(bs,h,1,ne,dk)).sum(-1 , keepdim = True)

		#beta_0 = beta_0.masked_fill(~beta_mask , float("-inf"))
		#beta_1 = beta_1.masked_fill(~beta_mask , float("-inf"))

		betas = tc.cat([beta_0 , beta_1] , dim = -1)
		betas = tc.softmax(betas , dim = -1)
		beta_0 , beta_1 = betas[:,:,:,:,0] , betas[:,:,:,:,1]
		
		R_Z = E_V.view(bs,h,ne,1,dk) * beta_0.view(bs,h,ne,ne,1) + E_V.view(bs,h,1,ne,dk) * beta_1.view(bs,h,ne,ne,1)
		#R_Z = E_V.view(bs,h,ne,1,dk) * 0.5 + E_V.view(bs,h,1,ne,dk) * 0.5

		R_Z = R_Z.masked_fill(~R_mas.expand(R_Z.size()).bool() , 0)
		E_Z = E_Z.masked_fill(~E_mas.expand(E_Z.size()).bool() , 0)

		R_Z = R_Z.view(bs,h,ne,ne,dk).permute(0,2,3,1,4).contiguous().view(bs,ne,ne,h*dk)
		E_Z = E_Z.view(bs,h,ne,dk).permute(0,2,1,3).contiguous().view(bs,ne,h*dk)

		return R_Z , E_Z

class FFN(nn.Module):
	def __init__(self , d_model , hidden_size = 1024):
		super().__init__()

		self.ln1 = nn.Linear(d_model , hidden_size)
		self.ln2 = nn.Linear(hidden_size , d_model)

		#self.reset_params()

	def reset_params(self):
		nn.init.xavier_normal_(self.ln1.weight.data)
		nn.init.xavier_normal_(self.ln2.weight.data)

		nn.init.constant_(self.ln1.bias.data , 0)
		nn.init.constant_(self.ln2.bias.data , 0)

	def forward(self , x , x_mas):
		x = F.relu(self.ln1(x))
		x = self.ln2(x)

		return x * x_mas

class Encoder_Layer(nn.Module):
	def __init__(self , h , d_model , hidden_size , dropout = 0.0):		
		super().__init__()

		assert d_model % h == 0

		self.d_model = d_model
		self.hidden_size = hidden_size

		self.att = Attention(h , d_model)
		self.lnorm_r_1 = nn.LayerNorm(d_model)
		self.lnorm_e_1 = nn.LayerNorm(d_model)
		self.drop_1 = nn.Dropout(dropout)

		self.ffn = FFN(d_model , hidden_size)
		self.lnorm_r_2 = nn.LayerNorm(d_model)
		self.lnorm_e_2 = nn.LayerNorm(d_model)
		self.drop_2 = nn.Dropout(dropout)


	def forward(self , R , E , R_mas , E_mas):		
		'''
			R: (bs , ne , ne , d)
			E: (bs , ne , d)

			R_mas: (bs , ne , ne , 1)
			E_mas: (bs , ne , 1)
		'''

		bs , ne , d = E.size()

		#-----attention-----

		R_Z , E_Z = self.att(R , E , R_mas , E_mas)
		R = self.lnorm_r_1(self.drop_1(R_Z) + R)
		E = self.lnorm_e_1(self.drop_1(E_Z) + E)


		#-----FFN-----
		R_Z , E_Z = self.ffn(R , R_mas) , self.ffn(E , E_mas)
		R = self.lnorm_r_2(self.drop_2(R_Z) + R)
		E = self.lnorm_e_2(self.drop_2(E_Z) + E)

		return R , E

class Encoder(nn.Module):
	def __init__(self , h = 8 , d_model = 768 , hidden_size = 2048 , num_layers = 6 , dropout = 0.0):
		super().__init__()

		self.d_model = d_model
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			Encoder_Layer(h , d_model , hidden_size , dropout = dropout)
			for _ in range(num_layers)
		])

	def forward(self , R , E , R_mas , E_mas):
		'''
			R: (bs , ne , ne , d)
			E: (bs , ne , d)
		'''

		bs , ne , d = E.size()
		assert d == self.d_model

		R_mas = R_mas.view(bs,ne,ne,1).float()
		E_mas = E_mas.view(bs,ne,1).float()
		R , E = R*R_mas , E*E_mas

		for layer in self.layers:
			R , E = layer(R , E , R_mas , E_mas)

		return R , E


	
