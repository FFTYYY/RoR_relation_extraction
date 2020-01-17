import pdb

class FakeLogger:
	def log(self , *pargs , **kwargs):
		pass


def get_data_and_rels(train_data_file, dataset_type, list_of_rel_files,
					  rel_weight_smooth=0, rel_weight_norm=False):
	'''
	train_data_file: str, a valid training file in XML format as semeval 2018
	dataset_type: str, it can be one of [“agenda”, “webnlg_sent”, “wiki_distant”]. 
		Currently, they adopt the same way to calculate rel_weight.
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

	relations , rel_weights = get_rel_weights(relations , dataset_type,
											  rel_weight_smooth, rel_weight_norm)

	return len(data) , relations , rel_weights

def file_content2data(file_content_xml, file_content_rel):
	'''
	file_content_xml: str, equivalent to “file_content_xml = open(xml_file).read()”
	file_content_rel: str, equivalent to “file_content_rel = open(rel_file).read()”

	return:
	data: list of dict, each dict is a data sample as you defined.
	'''
	from dataloader.dataloader_semeval_2018_task7 import parse_a_text_file , parse_a_key_file
	from dataloader.base import data_process

	data 				= parse_a_text_file(FakeLogger() , file_content_xml , dirty = True)
	data , rel_list 	= parse_a_key_file (FakeLogger() , data , file_content_rel)

	data = [d for _ , d in data.items()]

	return data_process(
		FakeLogger() , 
		data , [] , [] , rel_list , 
		"None" , 0 , False , False , 
	)[0]


def batch2loss(C, data, dataset_type, model ,
				loss_func,generator,
				freeze_model:bool=False, if_generate:bool=False):
	'''
	C: Namespace, where each attribute is accessed by “C.attr”. It means configurations
	data: a list of dict, each dict is a data sample as you loaded in “dataloader.py”
	dataset_type: str, it can be one of [“agenda”, “webnlg_sent”, “wiki_distant”]. They all use get_f1()
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
	from utils.train_util import get_data_from_batch
	import torch as tc
	from utils.scorer import get_f1

	if freeze_model:
		model = model.eval()
	else:
		model = model.train()

	sents , ents , anss , data_ent = get_data_from_batch(data , device = C.device)

	#----- get loss ----- # no need to backpropagate
	if freeze_model:
		with tc.no_grad():
			pred = model(sents, ents)
			loss = loss_func(pred, anss, ents)
	else:
		pred = model(sents, ents)
		loss = loss_func(pred, anss, ents)

	#----- generate -----

	if if_generate:
		ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
		generated = generator([pred] , data_ent , ans_rels = ans_rels)
	else:
		generated = ""

	#----- eval -----
	if dataset_type in ["agenda", "webnlg_sent", "wiki_distant"]:
		golden_content = C.test_rels if freeze_model else C.train_rels_1
		with open(golden_content , "r") as fil: golden_content = fil.read()

		f1_micro, f1_macro = get_f1(golden_content, generated , is_file_content=True)

	else:
		f1_micro , f1_macro = -1 , -1

	return {
			'performance': {
				'loss'		: loss , 
				'f1_micro'	: f1_micro , 
				'f1_macro'	: f1_macro , 
			},
			'generated': generated,
		}



def get_initializations(C):
	from torch.optim.lr_scheduler import LambdaLR
	import torch as tc
	from models import get_model 
	from main import initialize


	list_of_rel_files = [C.train_rels_1, C.train_rels_2, C.test_rels]
	train_data_len, relations, rel_weights = get_data_and_rels(
		C.train_text_1 , C.dataset , list_of_rel_files,
		C.rel_weight_smooth, C.rel_weight_norm
	)

	n_rel_typs , loss_func , generator = initialize(C , FakeLogger() , relations, rel_weights)

	model = get_model(C.model)(n_rel_typs=n_rel_typs,dropout=C.dropout)

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
	
	return model , optimizer , loss_func, generator ,  scheduler


def get_test_performance(
		test_file_content_xml , test_file_content_rel , 
		C , dataset_type , model , loss_func , generator , 
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
	from dataloader.base import get_rel_weights , get_file_content , data_process

	from dataloader.dataloader_semeval_2018_task7 import parse_a_text_file , parse_a_key_file

	test_data 				= parse_a_text_file(FakeLogger() , test_file_content_xml , dirty = True)
	test_data , rel_list 	= parse_a_key_file (FakeLogger() , test_data , test_file_content_rel)
	test_data = [d for _ , d in test_data.items()]
	test_data = data_process(
		FakeLogger() , 
		[] , test_data , [] , rel_list , 
		"None" , 0 , False , False , 
	)[1]

	f1_micro , f1_macro , loss , generated = test(C , FakeLogger() , 
		test_data , [model] , loss_func , generator , 
		mode = "test" , epoch_id = 0 , run_name = 0 , need_generated = True
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
		"semeval_2018_task7" , 
		["./data/semeval_2018_task7/1.1.relations.txt"] , 
	)
	print (len_data , relations , rel_weights)

	pdb.set_trace()

	#-----------------------------------------------------------------------------------------------



	train_data = file_content2data(
		open("./data/semeval_2018_task7/1.1.text.xml").read() , 
		open("./data/semeval_2018_task7/1.1.relations.txt").read() , 
	)
	print (len(train_data))
	pdb.set_trace()
	#-----------------------------------------------------------------------------------------------

	from config import get_config
	C , loger = get_config()
	model, optimizer, loss_func, generator , scheduler = get_initializations(C)
	pdb.set_trace()
	#-----------------------------------------------------------------------------------------------

	ret = batch2loss(C, train_data[:8], "agenda", model, optimizer, scheduler, loss_func , generator , 
				False, True)
	print (ret[0]["performance"])

	pdb.set_trace()
	#-----------------------------------------------------------------------------------------------

	result = get_test_performance(
		open("./data/semeval_2018_task7/2.test.text.xml").read() ,
		open("./data/semeval_2018_task7/keys.test.2.txt").read() ,
		C , "whatever" , model , loss_func , generator , 
	)
	print (result['performance'])
	pdb.set_trace()


if __name__ == "__main__":
	_test()
