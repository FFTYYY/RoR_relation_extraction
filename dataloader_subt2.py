from config import C
import os , sys
import os.path as path
import pdb
import xml.sax

class Entity:
	def __init__(self , start_pos , end_pos , name):
		self.s = start_pos
		self.e = end_pos

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

		self.ans = []

relations = set()

def parse_a_text_file(file_path):
	'''看起来还行'''
	with open(file_path , "r" , encoding = "utf-8") as fil:
		cont = fil.read()

	cont = cont.split("<doc>")[1].split("</doc>")[0] 				#去掉<doc>之前的和</doc>之后的内容 
	cont = filter(lambda x : x , cont.strip().split("</text>"))

	datas = {}
	for x in cont:

		x = x.strip()
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
		except Exception:
			pdb.set_trace()

		datas[text_id].ans.append(Relation(ent_a , ent_b , rel))
		relations.add(rel)

	return datas 

def run(data_path):

	train_name 		= "1.1.text.xml"
	test_name 		= "2.test.text.xml"
	train_key_name  = "1.1.relations.txt"
	test_key_name 	= "keys.test.2.txt"

	train_name 		= os.path.join(data_path , train_name)
	test_name 		= os.path.join(data_path , test_name)
	train_key_name 	= os.path.join(data_path , train_key_name)
	test_key_name 	= os.path.join(data_path , test_key_name)

	train_data 	= parse_a_text_file(train_name)
	test_data 	= parse_a_text_file(test_name)
	train_data 	= parse_a_key_file(train_data , train_key_name)
	test_data 	= parse_a_key_file(test_data , test_key_name)

	pdb.set_trace()

	return train_data


if __name__ == "__main__":
	run(data_path = C.data_path)