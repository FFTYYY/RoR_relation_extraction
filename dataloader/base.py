import fitlog
import numpy as np
from collections import Counter
from transformers import BertModel , BertTokenizer
import random
import pdb
#relations = ["COMPARE" , "MODEL-FEATURE" , "PART_WHOLE" , "RESULT" , "TOPIC" , "USAGE" , ]


class Entity:
	def __init__(self , start_pos , end_pos , name):
		self.s = start_pos
		self.e = end_pos
		self.name = name

class Relation:
	def __init__(self , ent_a , ent_b , type):
		self.u = ent_a
		self.v = ent_b
		self.type = type

class Data:
	def __init__(self , text_id = None , title = "" , abstract = "" , ents = []):
		self.text_id = text_id
		self.title = title.strip()
		self.abs = abstract.strip()
		self.ents = ents
		self.ans = []

		self.ent_names = [x.name for x in self.ents]

	def ent_name2id(self , ent):
		return self.ent_names.index(ent)
	def id2ent_name(self , ent_id):
		return self.ent_names[ent_id]

def cut(logger , data , dtype = "train"):

	if len(data.abs) >= 512:
		#print ("abs too long")
		data.abs = data.abs[:512]

	to_remove = []
	for x in data.ents:
		if x.s >= 512 or x.e >= 512:
			to_remove.append(x)

	if len(to_remove) > 0:
		if dtype == "test":
			for x in to_remove:
				data.ents.remove(x)
			logger.log ("Droped %d entity because too long in %s" % (len(to_remove) , dtype))

		elif dtype == "train":
			return None
		else:
			assert False

	if len(data.ents) > 50:
		if dtype != "test":
			return None


	return data

def bertize(logger , tokenizer , data):

	cont = data.abs
	tok_abs = ["[CLS]"] + tokenizer.tokenize(cont) + ["[SEP]"]

	for x in data.ents:
		guess_start = x.s - cont[:x.s].count(" ") + 5 #纯字母版本下的位置（+5是因为[CLS]）
		guess_end   = x.e - cont[:x.e].count(" ") + 5 #纯字母版本下的位置（+5是因为[CLS]）

		new_s = -1
		new_e = -1
		l = 0
		r = 0
		for i in range(len(tok_abs)):
			l = r
			r = l + len(tok_abs[i]) - tok_abs[i].count("##")*2
			if l <= guess_start and guess_start < r:
				new_s = i
			if l <= guess_end and guess_end < r:
				new_e = i

		try:
			assert new_s >= 0 and new_e >= 0
		except AssertionError:
			logger.log ("bad bertize")
			pdb.set_trace()

		old_s , x.s = x.s , new_s
		old_e , x.e = x.e , new_e

		if "".join(tok_abs[new_s : new_e]).replace("##" , "").lower() != cont[old_s : old_e].replace(" ","").lower():
			logger.log ("bad bertize")
			# pdb.set_trace()
			# TODO: skip special characters

	data.abs = tok_abs

	#pdb.set_trace()

	return data

def numberize(logger , tokenizer , data , relations):
	data.abs = tokenizer.convert_tokens_to_ids(data.abs)
	for x in data.ans:
		x.type = relations.index(x.type)
		x.u = data.ent_name2id(x.u)
		x.v = data.ent_name2id(x.v)
	return data


def get_file_content(file_path):
	with open(file_path , "r" , encoding = "utf-8") as fil:
		cont = fil.read()
	return cont

def get_rel_weights(rel_list , dataset_type , rel_weight_smooth = 0 , rel_weight_norm = False):
	rel_count = Counter(rel_list)
	relations = list(rel_count.keys())

	if dataset_type == "semeval_2018_task7":
		rel2wgh = {
			"NONE": 0 , "COMPARE": 1, "MODEL-FEATURE": 0.5, "PART_WHOLE": 0.5,
			"RESULT": 1, "TOPIC": 5, "USAGE": 0.5,
		}
		#rel2wgh = {
		#	"NONE": 1 , "COMPARE": 1, "MODEL-FEATURE": 1, "PART_WHOLE": 1,
		#	"RESULT": 1, "TOPIC": 1, "USAGE": 1,
		#}
		relations = ["COMPARE", "MODEL-FEATURE", "PART_WHOLE", "RESULT",
					 "TOPIC", "USAGE", "NONE"]
		rel_weights = [rel2wgh.get(r , 0.05) for r in relations]
	elif dataset_type == "ace_2005":
		rel2wgh = {
			"PART-WHOLE": 1, "PHYS":1, "GEN-AFF":1, "ORG-AFF":1, "ART":1, "PER-SOC":1, "NO_RELATION":0,
		}
		relations = ["PART-WHOLE", "PHYS", "GEN-AFF", "ORG-AFF", "ART", "PER-SOC", "NO_RELATION"]

		rel_weights = [float(rel2wgh[r]) for r in relations]
	else:
		rel_top_freq = rel_count.most_common(1)[0][-1]
		rel_weights = [(rel_top_freq + rel_weight_smooth) / (cnt + rel_weight_smooth) for cnt in rel_count.values()]
		if rel_weight_norm:
			rel_weights = np.array(rel_weights) / np.sum(rel_weights)
		
	return relations , rel_weights

def tokenize_and_index(logger , tokenizer , dataset , relations):
	for i , data in enumerate(dataset):
		dataset[i] = bertize(logger , tokenizer , data)
	for i , data in enumerate(dataset):
		dataset[i] = numberize(logger , tokenizer , data , relations)
	return dataset

def validize(logger , dataset , mode = "train"):
	'''
		for train mode , drop those too long , for test mode , assert no too long
	'''
	if mode == "train":
		to_rem = []
		for i , data in enumerate(dataset):
			got = cut(logger , data , "train")
			if got is not None:
				dataset[i] = got
			else:
				to_rem.append(i)
				logger.log ("*** droped one instance in train because too long")
		to_rem.reverse()
		for x in to_rem:
			dataset.pop(x)
	else:
		for i , data in enumerate(dataset):
			dataset[i] = cut(logger , data , "test")
			assert dataset[i] is not None # don't drop test instance

	return dataset


def data_process(
		C , logger , 
		train_data , test_data , valid_data , rel_list , 
		dataset_type , rel_weight_smooth , rel_weight_norm , verbose = True , 
	):

	#----- process relation list -----

	relations , rel_weights = get_rel_weights(rel_list , dataset_type)
	
	#----- post process -----
	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)


	def data_post_process(dataset , mode):
		dataset = tokenize_and_index(logger , tokenizer , dataset , relations)
		dataset = validize 			(logger , dataset , mode)
		return dataset

	train_data = data_post_process(train_data , "train")
	test_data  = data_post_process(test_data  , "test")
	valid_data = data_post_process(valid_data , "test")

	#----- special process ----

	# 如果 binary 和 pos_only 都开启，表示在进行测试，则不进行任何处理
	if C.binary and not C.pos_only:
		no_rel_idx = relations.index(C.no_rel_name)
		for data in [train_data , test_data , valid_data]:
			for x in data:
				for r in x.ans:
					r.type = (r.type != no_rel_idx) # 0 for negative , 1 for positive
		relations = ["NEGATIVE" , "POSITIVE"] 
		rel_weights = [1,1]
		C.no_rel_name = "NEGATIVE"
	if C.pos_only and not C.binary:
		no_rel_idx = relations.index(C.no_rel_name)

		#弹出所有标注的负例
		for data in [train_data , test_data , valid_data]:
			for x in data:
				to_rem = []
				for i , r in enumerate(x.ans):
					if r.type == no_rel_idx:
						to_rem.append(i)
				to_rem.reverse() # 倒过来弹出，防止影响顺序
				for i in to_rem:
					x.ans.pop(i)

		#弹出所有全负例的数据
		for data in [train_data , test_data , valid_data]:
			to_rem = []
			for i , x in enumerate(data):
				flag = False
				for r in x.ans:
					if r.type != no_rel_idx:
						flag = True
				if not flag: # all no_rel
					to_rem.append(i)
			to_rem.reverse() # 倒过来弹出，防止影响顺序
			for i in to_rem:
				data.pop(i)					

		relations.pop(no_rel_idx)
		rel_weights.pop(no_rel_idx)


	#----- final process ----

	random.shuffle(train_data)

	if verbose: 
		logger.log ("length of train / test / valid data = %d / %d / %d" % (
				len(train_data) , len(test_data) , len(valid_data) , 
		))

	return train_data , test_data , valid_data , relations , rel_weights


