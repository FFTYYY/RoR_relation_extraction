import sys
import torch as tc
import torch.nn as nn
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
		self.nemax = 20

		self.WQ = nn.Linear(self.dk , self.dk)
		self.WK = nn.Linear(self.dk , self.dk)
		self.WV = nn.Linear(self.dk , self.dk)

		self.relative_pos_emb = nn.Parameter( tc.zeros(2 * self.nemax, 2 * self.nemax, d_model) )
		#self.reset_params()

	def reset_params(self):
		nn.init.xavier_normal_(self.WQ.weight.data)
		nn.init.xavier_normal_(self.WK.weight.data)
		nn.init.xavier_normal_(self.WV.weight.data)

		nn.init.constant_(self.WQ.bias.data , 0)
		nn.init.constant_(self.WK.bias.data , 0)
		nn.init.constant_(self.WV.bias.data , 0)

	def forward(self , R , R_mas):
		'''
			R: (bs , ne , ne , d)
			R_mas: (bs , ne , ne , 1)
		'''

		h , dk = self.h , self.dk
		bs , ne , ne , d = R.size()
		assert d == self.d_model

		R = R.view(bs,ne,ne,h,dk).permute(0,3,1,2,4).contiguous() #(bs , h , ne , ne , dk)
		R_mas = R_mas.view(bs,1,ne,ne,1)

		Q , K , V = self.WQ(R) , self.WK(R) , self.WV(R)

		Q = Q.view(bs,h,ne*ne,dk)
		K = K.view(bs,h,ne*ne,dk)
		V = V.view(bs,h,ne*ne,dk)
		mas = R_mas.view(bs,1,ne*ne,1)
		att_mas = mas.view(bs,1,ne*ne,1) * mas.view(bs,1,1,ne*ne) # (bs,1,ne*ne,ne*ne)

		alpha = tc.matmul(Q , K.transpose(-1,-2))
		alpha = alpha - (1-att_mas)*100000 # mask for softmax
		alpha = tc.softmax(alpha / (dk ** 0.5) , dim = -1)

		R_Z = tc.matmul(alpha , V).view(bs,h,ne,ne,dk)

		R_Z = (R_Z * R_mas).permute(0,2,3,1,4).contiguous().view(bs,ne,ne,h*dk)

		return R_Z

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
		self.lnorm_1 = nn.LayerNorm(d_model)
		self.drop_1 = nn.Dropout(dropout)

		self.ffn = FFN(d_model , hidden_size)
		self.lnorm_2 = nn.LayerNorm(d_model)
		self.drop_2 = nn.Dropout(dropout)


	def forward(self , R , R_mas):
		'''
			R: (bs , ne , ne , d)
			R_mas: (bs , ne , ne , 1)
		'''

		#-----attention-----

		R_Z = self.att(R , R_mas)
		R = self.lnorm_1(self.drop_1(R_Z) + R)


		#-----FFN-----
		R_Z = self.ffn(R , R_mas)
		R = self.lnorm_2(self.drop_2(R_Z) + R)

		return R

class Encoder(nn.Module):
	def __init__(self , h = 8 , d_model = 768 , hidden_size = 2048 , num_layers = 6 , dropout = 0.0, device = 0):
		super().__init__()

		self.nemax = 50
		self.d_model = d_model
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			Encoder_Layer(h , d_model , hidden_size , dropout = dropout)
			for _ in range(num_layers)
		])

		self.row_emb = nn.Parameter( tc.zeros(self.nemax , device = device) )
		self.col_emb = nn.Parameter( tc.zeros(self.nemax , device = device) )
		self.reset_params()

	def reset_params(self):
		nn.init.normal_(self.row_emb.data , 0 , 1e-4)
		nn.init.normal_(self.col_emb.data , 0 , 1e-4)

	def forward(self , R , R_mas):
		'''
			R: (bs , ne , ne , d)
			sent_enc: (bs , n , d)

		'''

		bs , ne , ne , d = R.size()
		assert d == self.d_model

		R_mas = R_mas.view(bs,ne,ne,1).float()

		R = R + self.row_emb[:ne].view(1,ne,1,1) + self.col_emb[:ne].view(1,1,ne,1)

		for layer in self.layers:
			R = layer(R , R_mas)

		return R


	
