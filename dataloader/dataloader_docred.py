'''没写完'''

import os , sys
import numpy as np
import pdb
import random
import fitlog
from base import *
import json

fitlog.debug()
fitlog.commit(__file__)

def parse_a_file(text):
	c = json.loads(text)

	datas = []
	for i , x in enumerate(c):
		title = x["title"]

		vs = x["vertexSet"]
		ents  = [x["vertexSet"][i][j] for i in range(len(x["vertexSet"])) for j in range(len(x["vertexSet"][i]))] 

		pass
		#TODO


def file_content2data(
		logger , 
		train_text  , test_text ,  valid_text ,  
		dataset_type , rel_weight_smooth , rel_weight_norm , verbose = True , 
	):

	#pdb.set_trace()
	a = json.loads(train_text)


	return None
	#return data_process(
	#	logger , 
	#	train_data , test_data , valid_data , rel_list , 
	#	dataset_type , rel_weight_smooth , rel_weight_norm , verbose
	#)

def read_data(
		logger , 
		file_train_text , 
		file_test_text ,
		file_valid_text ,
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	):

	train_text = get_file_content(file_train_text)
	test_text = get_file_content(file_test_text)
	valid_text = get_file_content(file_valid_text)
	return file_content2data(
		logger , 
		train_text , test_text , valid_text ,
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	)


if __name__ == "__main__":

	read_data(
		None , 
		"../data/docred/train_annotated.json" , 
		"../data/docred/test.json" , 
		"../data/docred/dev.json" , 
		"docred" , False , False , 
	)