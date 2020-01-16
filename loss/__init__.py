from .losses import loss_1 , loss_2 , loss_3 , loss_4

class LossFunction:
	def __init__(self , inner_func , no_rel , class_weight = [1,0.5,0.5,1,5,0.5,1]):
		self.no_rel = no_rel
		self.class_weight = class_weight

		self.inner_func = inner_func

	def __call__(self , pred , anss , ents):
		return self.inner_func(pred , anss , ents , self.no_rel , self.class_weight)

def get_loss_func(name , no_rel , class_weight):
	inner_func = {
		"loss_1" : loss_1 , 
		"loss_2" : loss_2 , 
		"loss_3" : loss_3 , 
		"loss_4" : loss_4 , 
	}[name]

	return LossFunction(inner_func , no_rel , class_weight)

