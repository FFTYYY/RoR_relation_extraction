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
	def __init__(self , text_id , title , abstract , ents):
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

def data_process(
		logger , 
		train_data , test_data , valid_data , rel_list , 
		dataset_type , rel_weight_smooth , rel_weight_norm , verbose = True , 
	):

	#----- process relation list -----

	rel_count = Counter(rel_list)
	relations = list(rel_count.keys())

	if dataset_type == 'semeval_2018_task7':
		rel2wgh = {
			"COMPARE": 1, "MODEL-FEATURE": 0.5, "PART_WHOLE": 0.5,
			"RESULT": 1, "TOPIC": 5, "USAGE": 0.5,
		}
		relations = ["COMPARE", "MODEL-FEATURE", "PART_WHOLE", "RESULT",
					 "TOPIC", "USAGE", ]
		rel_weights = [rel2wgh[r] for r in relations]
	else:
		rel_top_freq = rel_count.most_common(1)[0][-1]
		rel_weights = [(rel_top_freq + rel_weight_smooth) / (cnt + rel_weight_smooth) for cnt in rel_count.values()]
		if rel_weight_norm:
			rel_weights = np.array(rel_weights) / np.sum(rel_weights)

	#----- tokenize & index -----
	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	for i , data in enumerate(train_data):
		train_data[i] = bertize(logger , tokenizer , data)
	for i , data in  enumerate(test_data):
		test_data [i] = bertize(logger , tokenizer , data)
	for i , data in enumerate(valid_data):
		valid_data[i] = bertize(logger , tokenizer , data)

	for i , data in enumerate(train_data):
		train_data[i] = numberize(logger , tokenizer , data , relations)
	for i , data in  enumerate(test_data):
		test_data [i] = numberize(logger , tokenizer , data , relations)
	for i , data in enumerate(valid_data):
		valid_data[i] = numberize(logger , tokenizer , data , relations)

	#----- drop those who are too long to feed into bert ----
	to_rem = []
	for i , data in enumerate(train_data):
		got = cut(logger , data , "train")
		if got is not None:
			train_data[i] = got
		else:
			to_rem.append(i)
			logger.log ("*** droped one instance in train because too long")
	to_rem.reverse()
	for x in to_rem:
		train_data.pop(x)

	for i , data in enumerate(test_data):
		test_data[i] = cut(logger , data , "test")
		assert test_data[i] is not None # don't drop test instance

	for i , data in enumerate(valid_data):
		valid_data[i] = cut(logger , data , "test")
		assert valid_data[i] is not None # also don't drop valid instance


	#----- final process ----


	random.shuffle(train_data)

	if verbose: 
		logger.log ("length of train / test / valid data = %d / %d / %d" % (
				len(train_data) , len(test_data) , len(valid_data) , 
		))

	return train_data , test_data , valid_data , relations , rel_weights


