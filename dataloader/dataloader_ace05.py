import os , sys
import numpy as np
import pdb
import random
import fitlog
from .base import *
import json

#fitlog.commit(__file__)

def parse_a_file(logger , file_text):

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

		if not e1_name in datas[text].ent_names:
			datas[text].ents.append(Entity(e1_s , e1_e , e1_name))
			datas[text].ent_names.append(e1_name)

		if not e2_name in datas[text].ent_names:
			datas[text].ents.append(Entity(e2_s , e2_e , e2_name))
			datas[text].ent_names.append(e2_name)

		datas[text].ans .append(Relation(e1_name , e2_name , type = x["relLabels"][0]))
		rel_list.append(x["relLabels"][0])

	datas = [datas[x] for x in datas]

	return datas , rel_list

def _read_data(
		logger , file_train_text , file_test_text , file_valid_text ,
		dataset_type , rel_weight_smooth , rel_weight_norm ,
	):

	train_data , rel_list_1 = parse_a_file(logger , file_train_text)
	test_data  , rel_list_2 = parse_a_file(logger , file_test_text)
	valid_data , rel_list_3 = parse_a_file(logger , file_valid_text)

	rel_list = rel_list_1 + rel_list_2 + rel_list_3


	return data_process(
		logger , 
		train_data , test_data , valid_data , rel_list , 
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	)


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