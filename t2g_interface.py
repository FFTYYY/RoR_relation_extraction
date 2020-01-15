
class FakeLogger:
	def __get__(self , *pargs , **kwargs):
		pass


def get_data_and_rels(train_data_file, list_of_rel_files , dataset_type = "None"):
	'''
	train_data_file: str, a valid training file in XML format as semeval 2018
	list_of_rel_files: a list of str, each str is a valid file name storing relations

	return:
	train_data_size: int, number of data samples in “train_data_file”
	relations: a list of str, each str is the relation. The relations list include all the relations in the list_of_rel_files
	relation_weights: a list of float, each float is the weight of the corresponding relation in “relations”

	requirement: I wish this function to be fast, so please skip time-consuming actions such as “bertize()”
	'''
	from dataloader.base import get_rel_weights , get_file_content
	from dataloader.dataloader_semeval_2018_task7 import parse_a_text_file , parse_a_key_file

	train_data_file = get_file_content(train_data_file)

	data = parse_a_text_file(FakeLogger() , train_data_file , dirty = False)
	relations = []
	for rel_file in list_of_rel_files:
		rel_file  = get_file_content(rel_file)
		data , rel_list = parse_a_key_file (FakeLogger() , data , rel_file)
		relations += rel_list

	#make datas list
	data = [d for _ , d in data.items()]

	relations , rel_weights = get_rel_weights(relations , dataset_type)

	return len(data) , relations , rel_weights

def file_content2data(file_content_xml, file_content_rel):
	'''
	file_content_xml: str, equivalent to “file_content_xml = open(xml_file).read()”
	file_content_rel: str, equivalent to “file_content_rel = open(rel_file).read()”

	return:
	data: list of dict, each dict is a data sample as you defined.
	'''
	from dataloader.dataloader_semeval_2018_task7 import file_content2data as file2data


	train_data , _ , _ , _ , _ = file2data(
		FakeLogger() , 
		file_content_xml , file_content_rel , 
		"" , "" ,
		"" , "" , 
		"" , "" , 
		"None" , 0 , False , verbose = False
	)

	return train_data


def batch2loss(C, data, model, optimizer, scheduler, loss_func,
				no_rel, rel_weights, relations,
				freeze_model:bool=False, if_generate:bool=False):
	'''
	C: Namespace, where each attribute is accessed by “C.attr”. It means configurations
	data: a list of dict, each dict is a data sample as you loaded in “dataloader.py”
	model: the model which is already model.to(device)
	freeze_model: bool, if in evaluation mode, then True; if in training mode, then False
	generate: bool, if generated txt is needed

	return:
	result: dict, {
	'performance': {'loss': avg_loss: float=0., 'f1_micro': f1_micro: float=-1,
					'f1_macro': f1_macro: float=-1},
	'generated': generated: str,
	}
	model: the same as the input model, but with parameters updated by optimizer
	optimizer: the same as the input, but updated
	scheduler: the same as the input, but updated
	'''
	from test import get_output
	from train import update_batch , get_evaluate
	from generate import generate
	from utils.train_util import get_data_from_batch
	import torch as tc

	if freeze_model:
		model = model.eval()
	else:
		model = model.train()

	sents , ents , anss , data_ent = get_data_from_batch(data , device = C.device)

	#----- get loss -----
	if freeze_model:
		with tc.no_grad():
			model , preds , loss , partial_generated = get_output(
				C,FakeLogger(),no_rel,rel_weights,relations,[model],C.device,loss_func,sents,ents,anss,data_ent
			)
		pred = preds[0]
	else:
		loss , pred = update_batch(
			C,FakeLogger(),no_rel,model,optimizer,scheduler,loss_func,sents,ents,anss,data_ent
		)

	#----- generate -----

	if if_generate:
		ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
		generated = generate([pred] , data_ent , relations , no_rel , ans_rels = ans_rels)
	else:
		generated = ""

	#----- eval -----
	pass

	return (
		{
			'performance': {
				'loss'		: loss , 
				'f1_micro'	: -1 , 
				'f1_macro'	: -1 , 
			},
			'generated': generated,
		} , 
		model , optimizer , scheduler , 
	)



def get_initializations(C):
	from torch.optim.lr_scheduler import LambdaLR
	from train import tc, get_data_and_rels, models, loss_funcs

	list_of_rel_files = [C.train_rels_1, C.train_rels_2, C.test_rels]
	train_data_len, _, relations, rel_weights = \
		get_data_and_rels(C.train_text_1, list_of_rel_files)

	if C.rel_only:
		no_rel = -1
	else:
		rel_weights += [C.no_rel_weight]
		no_rel = len(relations)

	n_rel_typs = len(rel_weights)

	model = models[C.model](n_rel_typs=n_rel_typs,dropout=C.dropout).to(C.device)

	optimizer = tc.optim.Adam(params=model.parameters(),lr=C.t2g_lr)
	batch_numb = (train_data_len // C.t2g_batch_size) + int(
		(train_data_len % C.t2g_batch_size) != 0)
	num_training_steps = C.epoch_numb * batch_numb
	
	class _LRPolicy:
		""" Create a schedule with a learning rate that decreases following the
		values of the cosine function between 0 and `pi * cycles` after a warmup
		period during which it increases linearly between 0 and 1.
		"""
	
		def __init__(self, num_warmup_steps, num_training_steps,
					num_cycles=0.5):
			self.num_warmup_steps = num_warmup_steps
			self.num_training_steps = num_training_steps
			self.num_cycles = num_cycles
	
		def __call__(self, current_step):
			import math
			num_warmup_steps = self.num_warmup_steps
			num_training_steps = self.num_training_steps
			num_cycles = self.num_cycles
	
			if current_step < num_warmup_steps:
				return float(current_step) / float(max(1, num_warmup_steps))
			progress = float(current_step - num_warmup_steps) / float(
				max(1, num_training_steps - num_warmup_steps))
			return max(0.0, 0.5 * (1.0 + math.cos(
				math.pi * float(num_cycles) * 2.0 * progress)))

	scheduler = LambdaLR( # using _LRPolicy is for making it pickle-able.
		optimizer, _LRPolicy(C.n_warmup, num_training_steps))
	loss_func = loss_funcs[C.loss]
	
	return model, optimizer, loss_func, scheduler, \
		relations, rel_weights, no_rel, n_rel_typs


def get_test_performance(
		test_file_content_xml , test_file_content_rel , 
		C , model , relations , rel_weights , no_rel , 
	):
	'''
	# we can discuss this interface
	# the main concern is that we cannot throw away sentences longer than 512 for test set, so we must think of some work-arounds

	return 
	result: dict, {
		'performance': {'loss': avg_loss: float=0., 'f1_micro': f1_micro: float=-1,
						'f1_macro': f1_macro: float=-1},
		'generated': generated: str,
	}
	'''

	# 先暂时不管『we cannot throw away sentences longer than 512 for test set』

	from test import test
	from dataloader.dataloader_semeval_2018_task7 import file_content2data as file2data

	_ , test_data , _ , _ , _ = file2data(
		FakeLogger() , 
		"" , "" , 
		"" , "" ,
		test_file_content_xml , test_file_content_rel , 
		"" , "" , 
		"None" , 0 , False , verbose = False
	)

	f1_micro , f1_macro , loss , generated = test(C , FakeLogger() , 
		test_data , [model] , 
		relations , rel_weights , no_rel , 
		mode = "valid" , epoch_id = 0 , ensemble_id = 0 , need_generated = True
	)

	return {
		'performance': {
			'loss': loss , 
			'f1_micro': f1_micro ,
			'f1_macro': f1_macro , 
		},
		'generated': generated,
	}


def _test():
	import pdb

	#-----------------------------------------------------------------------------------------------
	len_data , relations , rel_weights = get_data_and_rels(
		"./data/semeval_2018_task7/1.1.text.xml" ,
		["./data/semeval_2018_task7/keys.test.1.1.txt"] , 
		dataset_type = "semeval_2018_task7" , 
	)
	print (len_data , relations , rel_weights)

	pdb.set_trace()

	#-----------------------------------------------------------------------------------------------

	train_data = file_content2data(
		"./data/semeval_2018_task7/1.1.text.xml" ,
		"./data/semeval_2018_task7/keys.test.1.1.txt"
	)
	print (len(train_data))
	pdb.set_trace()
	#-----------------------------------------------------------------------------------------------

	from config import get_config
	C = get_config()
	model, optimizer, loss_func, scheduler, relations, rel_weights, no_rel, n_rel_typs = get_initializations(C)
	pdb.set_trace()

	#-----------------------------------------------------------------------------------------------

	result = get_test_performance(
		"./data/semeval_2018_task7/2.test.text.xml" ,
		"./data/semeval_2018_task7/keys.test.2.txt" ,
		C , model , relations , rel_weights , no_rel , 
	)
	print (result)
	pdb.set_trace()


if __name__ == "__main__":
	_test()
