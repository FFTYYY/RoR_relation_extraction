import os , sys
import numpy as np
import pdb
import random
import fitlog
from .base import *

#fitlog.commit(__file__)


def parse_a_text_file(logger , cont , dirty = False):
	'''看起来还行'''

	cont = cont.split("<doc>")[1].split("</doc>")[0] 				#去掉<doc>之前的和</doc>之后的内容
	cont = filter(lambda x : x , cont.strip().split("</text>"))

	datas = {}
	for x in cont:
		try:
			x = x.strip().replace("\n" , " ").replace("\t" , " ")
			x = x[10:] #remove  <text id=\"

			assert '"' in x, "the quote sign is not in text string"
			text_id, x = x.split('"', 1)
			x = '"' + x

			#text_id , x = x[:8] , x[8:] #text id 一定是XXX-XXXX

			x = x.split("<title>")[1].strip() #remove <title>
			title , x = x.split("</title>")
			while "</entity>" in title: #remove entity in title
				title = title.split("<entity id=\"" , 1)[0] + " " + title.split("\">" , 1)[1]
				title = title.replace("</entity>" , "" , 1)
			title = " ".join(list(filter(lambda x : x , title.strip().split(" ")))) #去掉多余空格

			x = x.split("<abstract>")[1].strip().split("</abstract>")[0].strip() #remove <abstract> , <\abstract>

			ents = []
			while "</entity>" in x:
				ent_id = x.split("<entity id=\"" , 1)[1].split("\">" , 1)[0]
				ent_head = "<entity id=\"%s\">" % ent_id

				ent_start = x.find(ent_head)
				x = x.replace(ent_head , "" , 1)
				ent_end = x.find("</entity>")
				x = x.replace("</entity>" , "" , 1)

				while x[ent_start] == " ": #实体名带空格
					ent_start += 1
				while x[ent_end-1] == " ": #实体名带空格
					ent_end -= 1

				if not (ent_start == 0 or x[ent_start-1] == " "): #实体名在一串字符中间
					x = x[ : ent_start] + " " + x[ent_start : ]
					ent_start += 1
					ent_end += 1

				if not (ent_end == len(x) or x[ent_end] == " "): #实体名在一串字符中间
					x = x[ : ent_end] + " " + x[ent_end : ]

				assert ent_start >= 0 and ent_end >= 0

				ents.append(Entity(ent_start , ent_end , name = ent_id))

			assert x.find("<entity") < 0 and x.find("</entity>") < 0
			assert len(ents) > 0

			rem_ent = []
			for e in ents:
				try:
					assert (e.s == 0 or x[e.s-1] == " ") and x[e.s] != " "
					assert (e.e == len(x) or x[e.e] == " ")  and x[e.e-1] != " "
				except AssertionError:
					rem_ent.append(e)
				except IndexError:
					if dirty:
						assert False  #直接扔掉这实例
					rem_ent.append(e)

			if len(rem_ent) > 0:
				if dirty:
					assert False #直接扔掉这实例

				logger.log ("Bad entity showed. in %s" % e.name)
				for e in rem_ent:
					#drop that
					ents.remove(e)

			abstract = x

			datas[text_id] = Data(
				text_id 	= text_id ,
				title 		= title ,
				abstract 	= abstract ,
				ents 		= ents ,
			)
		except Exception:
			# if error occured , give up this data sample.
			if not dirty: #but not for clean data
				pdb.set_trace()

	return datas

def parse_a_key_file(logger , datas , cont , dtype = "test"):
	relations = []

	cont = cont.strip().split("\n")

	for x in cont:
		rel , x = x.strip().split("(")
		x = x.split(")")[0] #去掉末尾)
		next_ = x.split(",")

		if len(next_) == 3:
			assert next_[2] == "REVERSE"
			ent_b , ent_a = next_[:2]
		else:
			ent_a , ent_b = next_[:2]

		text_id = ent_a.split(".")[0]

		try:
			assert ent_b.split(".")[0] == text_id
		except AssertionError:
			pdb.set_trace()

		if datas.get(text_id) is None:
			# data dropped
			continue

		if not ent_a in datas[text_id].ent_names or not ent_b in datas[text_id].ent_names:
			# entity not in data (entity dropped in title)
			continue

		datas[text_id].ans.append(Relation(ent_a , ent_b , rel))

		if dtype == "train":
			if rel == "COMPARE":
				datas[text_id].ans.append(Relation(ent_b , ent_a , rel))


		relations.append(rel)

	return datas, relations


def file_content2data(
		C , logger , 
		train_text_1 , train_rels_1 , 
		train_text_2 , train_rels_2 ,
		test_text , test_rels , 
		valid_text , valid_rels , 
		dataset_type , rel_weight_smooth , rel_weight_norm , verbose = True
	):

	#----- read data files -----
	train_data_1 				= parse_a_text_file(logger , train_text_1 , dirty = False)
	train_data_1 , rel_list1 	= parse_a_key_file (logger , train_data_1 , train_rels_1 , dtype = "train")

	train_data_2 				= parse_a_text_file(logger , train_text_2 , dirty = True)
	train_data_2 , rel_list2 	= parse_a_key_file (logger , train_data_2 , train_rels_2 , dtype = "train")
	train_data = train_data_1
	train_data.update(train_data_2)

	test_data 				= parse_a_text_file(logger , test_text)
	test_data  , rel_list3 	= parse_a_key_file (logger , test_data , test_rels , dtype = "test")

	valid_data 				= parse_a_text_file(logger , valid_text)
	valid_data , rel_list4 	= parse_a_key_file (logger , valid_data , valid_rels , dtype = "test")

	rel_list = rel_list1 + rel_list2 + rel_list3 + rel_list4

	#make datas list
	train_data = [d for _ , d in train_data.items()]
	test_data  = [d for _ , d in  test_data.items()]
	valid_data = [d for _ , d in valid_data.items()]

	return data_process(
		C , logger , 
		train_data , test_data , valid_data , rel_list , 
		dataset_type , rel_weight_smooth , rel_weight_norm , verbose , 
	)

def read_data(
		C , logger , 
		file_train_text_1 , file_train_rels_1 , 
		file_train_text_2 , file_train_rels_2 , 
		file_test_text , file_test_rels , 
		file_valid_text , file_valid_rels , 
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	):

	train_text_1 = get_file_content(file_train_text_1)
	train_rels_1 = get_file_content(file_train_rels_1)
	train_text_2 = get_file_content(file_train_text_2)
	train_rels_2 = get_file_content(file_train_rels_2)
	test_text = get_file_content(file_test_text)
	test_rels = get_file_content(file_test_rels)
	valid_text = get_file_content(file_valid_text)
	valid_rels = get_file_content(file_valid_rels)
	return file_content2data(
		C , logger , 
		train_text_1 , train_rels_1 , 
		train_text_2 , train_rels_2 , 
		test_text , test_rels , 
		valid_text , valid_rels , 
		dataset_type , rel_weight_smooth , rel_weight_norm , 
	)


if __name__ == "__main__":
	from config import C

	read_data(C.train_text , C.train_rels , C.test_text , C.test_rels)