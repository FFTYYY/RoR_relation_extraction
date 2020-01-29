import os , sys
import numpy as np
import pdb
import random
import fitlog
from .base import *
import json

#fitlog.commit(__file__)

def parse_a_file(logger , file_text , data_name = "test"):

	datas = {}

	with open(file_text , "r" , encoding = "utf-8") as fil:
		cont = fil.read()
	cont = cont.replace("}\n" , "},\n")
	cont = cont.strip()[:-1]
	cont = "[\n" + cont + "\n]\n"
	cont = json.loads(cont)

	rel_list = []

	for x in cont:
		text = " ".join(x["words"])
		nepairs = json.loads(x["nePairs"])[0]


		rel_type , rel_direct = x["relLabels"][0].split("(")
		rel_direct = "(" + rel_direct

		#no no_rel in train, but yes in valid/test because generating need
		if data_name == "train":
			if rel_type == "NO_RELATION":
				continue

		e1_s , e1_e , e1_name = nepairs["m1"]["start"] , nepairs["m1"]["end"] , nepairs["m1"]["id"]
		e2_s , e2_e , e2_name = nepairs["m2"]["start"] , nepairs["m2"]["end"] , nepairs["m2"]["id"]
		e1_s = len(" ".join(x["words"][:e1_s])) + (e1_s!=0) # +1 for space
		e1_e = len(" ".join(x["words"][:e1_e])) 
		e2_s = len(" ".join(x["words"][:e2_s])) + (e2_s!=0) # +1 for space
		e2_e = len(" ".join(x["words"][:e2_e])) 

		try:
			assert text[e1_s : e1_e] == " ".join(x["words"][nepairs["m1"]["start"] : nepairs["m1"]["end"]])
			assert text[e2_s : e2_e] == " ".join(x["words"][nepairs["m2"]["start"] : nepairs["m2"]["end"]])
		except AssertionError:
			pdb.set_trace()


		try:
			assert len(x["relLabels"]) == 1
			assert len(json.loads(x["nePairs"])) == 1
		except AssertionError:
			pdb.set_trace()

		if datas.get(text) is None:
			datas[text] = Data(abstract = text , ents = [])
			datas[text].text_id = "%s_text_%d" % (data_name , len(datas))
			datas[text].fake_ent_names = []

		if not e1_name in datas[text].fake_ent_names:
			e1_real_name = "%s.%d" % (datas[text].text_id , len(datas[text].fake_ent_names)+1) #(text_id.编号)
			datas[text].ents.append(Entity(e1_s , e1_e , e1_real_name))
			datas[text].ent_names.append(e1_real_name)
			datas[text].fake_ent_names.append(e1_name)
		e1_idx = datas[text].fake_ent_names.index(e1_name)+1
		e1_real_name = "%s.%d" % (datas[text].text_id , e1_idx)

		if not e2_name in datas[text].fake_ent_names:
			e2_real_name = "%s.%d" % (datas[text].text_id , len(datas[text].fake_ent_names)+1) #(text_id.编号)
			datas[text].ents.append(Entity(e2_s , e2_e , e2_real_name))
			datas[text].ent_names.append(e2_real_name)
			datas[text].fake_ent_names.append(e2_name)
		e2_idx = datas[text].fake_ent_names.index(e2_name)+1
		e2_real_name = "%s.%d" % (datas[text].text_id , e2_idx)

		if rel_direct == "(Arg-1,Arg-2)":
			datas[text].ans.append(Relation(e1_real_name , e2_real_name , type = rel_type))
		elif rel_direct == "(Arg-2,Arg-1)":
			datas[text].ans.append(Relation(e2_real_name , e1_real_name , type = rel_type))
		elif rel_direct == "(Arg-1,Arg-1)": # 双向关系
			#只添加正向边
			if e1_idx >= e2_idx: #确保小到大
				e1_real_name , e2_real_name = e2_real_name , e1_real_name
			datas[text].ans.append(Relation(e1_real_name , e2_real_name , type = rel_type))

			#if data_name == "train": # 在训练集中，也添加反向边
			#	datas[text].ans.append(Relation(e2_real_name , e1_real_name , type = rel_type))

		else:
			assert False


		rel_list.append(rel_type)


	datas = [datas[x] for x in datas]

	return datas , rel_list

def _read_data(
		logger , file_train_text , file_test_text , file_valid_text ,
		dataset_type , rel_weight_smooth , rel_weight_norm ,
	):

	train_data , rel_list_1 = parse_a_file(logger , file_train_text , "train")
	test_data  , rel_list_2 = parse_a_file(logger , file_test_text  , "test" )
	valid_data , rel_list_3 = parse_a_file(logger , file_valid_text , "valid")

	rel_list = rel_list_1 + rel_list_2 + rel_list_3


	train_data , test_data , valid_data , relations , rel_weights = data_process(
		logger , 
		train_data , test_data , valid_data , rel_list , 
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	)

	#'''
	valid_cont = [x.abs for x in valid_data] + [x.abs for x in test_data]
	_to_rem = []
	for i , x in enumerate(train_data): #drop those valid in train
		if x.abs in valid_cont:
			_to_rem.append(i)
			logger.log("oh, found overlap in train and test/dev")
	_to_rem.reverse()
	for i in _to_rem:
		train_data.pop(i)
	#'''

	return train_data , test_data , valid_data , relations , rel_weights


def read_data(
		logger , 
		train_text_1 , train_rels_1 ,
		train_text_2 , train_rels_2 ,
		test_text  , test_rels ,
		valid_text , valid_rels ,
		dataset_type, rel_weight_smooth, rel_weight_norm,
	):

	return _read_data(
		logger , train_text_1 , test_text , valid_text ,
		dataset_type , rel_weight_smooth , rel_weight_norm ,
	)
	
if __name__ == "__main__":
	from config import C , logger

	read_data(
		logger , C.train_text_1 , C.test_text , C.test_rels , 
		"ace05" , C.rel_weight_smooth , C.rel_weight_norm
	)