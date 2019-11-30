from config import C
import os , sys
import os.path as path
import pdb
from transformers import BertModel , BertTokenizer


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



relations = set()
def rel2id(rel):
	return relations.index(rel)
def id2rel(i):
	return relations[i]

def parse_a_text_file(file_path):
	'''看起来还行'''
	with open(file_path , "r" , encoding = "utf-8") as fil:
		cont = fil.read()

	cont = cont.split("<doc>")[1].split("</doc>")[0] 				#去掉<doc>之前的和</doc>之后的内容 
	cont = filter(lambda x : x , cont.strip().split("</text>"))

	datas = {}
	for x in cont:

		x = x.strip().replace("\n" , " ").replace("\t" , " ")
		x = x[10:] #remove  <text id=\"
		text_id , x = x[:8] , x[8:] #text id 一定是XXX-XXXX

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

		for e in ents:
			assert (e.s == 0 or x[e.s-1] == " ") and x[e.s] != " "
			assert (e.e == len(x) or x[e.e] == " ")  and x[e.e-1] != " "

		abstract = x

		datas[text_id] = Data(
			text_id 	= text_id ,  
			title 		= title ,  
			abstract 	= abstract ,  
			ents 		= ents ,  
		)

	return datas

def parse_a_key_file(datas , file_path):

	with open(file_path , "r" , encoding = "utf-8") as fil:
		cont = fil.read()
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

		datas[text_id].ans.append(Relation(ent_a , ent_b , rel))
		relations.add(rel)

	return datas 

bert_type = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_type)
def bertize(data):

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
			print ("bad bertize")
			pdb.set_trace()

		old_s , x.s = x.s , new_s
		old_e , x.e = x.e , new_e

		try:
			assert "".join(tok_abs[new_s : new_e]).replace("##" , "").lower() == cont[old_s : old_e].replace(" ","").lower()
		except AssertionError:
			print ("bad bertize")
			pdb.set_trace()

	data.abs = tok_abs

	#pdb.set_trace()

	return data

def numberize(data):
	data.abs = tokenizer.convert_tokens_to_ids(data.abs)
	for x in data.ans:
		x.type = rel2id(x.type)
		x.u = data.ent_name2id(x.u)
		x.v = data.ent_name2id(x.v)
	return data

_cnt = 0
def cut(data , dtype = "train"):
	global _cnt
	if len(data.abs) >= 512:
		#print ("abs too long")
		data.abs = data.abs[:512]

	to_remove = []
	for x in data.ents:
		if x.s >= 512 or x.e >= 512:
			#pdb.set_trace()
			print ("Droped one entity because too long in %s" % dtype)
			to_remove.append(x)
			#return None
			#assert False
	for x in to_remove:
		data.ents.remove(x)
	return data

def run(train_text , train_rels , test_text , test_rels):

	global relations

	train_data 	= parse_a_text_file(train_text)
	test_data 	= parse_a_text_file(test_text)
	train_data 	= parse_a_key_file(train_data , train_rels)
	test_data 	= parse_a_key_file(test_data , test_rels)

	for name , data in train_data.items():
		train_data[name] = bertize(data)
	for name , data in test_data.items():
		test_data[name]  = bertize(data)

	relations = list(relations)

	for name , data in train_data.items():
		train_data[name] = numberize(data)
	for name , data in test_data.items():
		test_data[name]  = numberize(data)

	for name , data in train_data.items():
		got = cut(data , "train")
		if got is not None:
			train_data[name] = got			
		else:
			print ("*** droped one instance in train because too long")

	for name , data in test_data.items():
		got = cut(data , "test")
		if got is not None:
			test_data[name] = got
		else:
			print ("*** droped one instance in test because too long")


	#listize
	train_data = [d for _ , d in train_data.items()]
	test_data = [d for _ , d in test_data.items()]


	return train_data , test_data


if __name__ == "__main__":
	run(C.train_text , C.train_rels , C.test_text , C.test_rels)