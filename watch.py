'''
	watch trained model results
'''
from transformers import BertModel , BertTokenizer
from dataloader import read_data
from tqdm import tqdm
import torch as tc
from models import models
import pdb
import os , sys
import math
from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup
from loss import loss_funcs
from generate import generate
from test import test
from utils.train_util import pad_sents , get_data_from_batch
from utils.scorer import get_f1
import fitlog
import pickle
import re
from YTools.universe.beautiful_str import beautiful_str
import json

fitlog.debug()
fitlog.commit(__file__)

def load_data(C , logger):
	data_train , data_test , data_valid , relations, rel_weights = read_data(C.dataset)(
		logger , 
		C.train_text_1 , C.train_rels_1 ,
		C.train_text_2 , C.train_rels_2 ,
		C.test_text  , C.test_rels ,
		C.valid_text , C.valid_rels ,
		C.dataset, C.rel_weight_smooth, C.rel_weight_norm,
	)

	return data_train , data_test , data_valid , relations, rel_weights

def generate_output(
		C , logger , 
		dataset , models , 
		relations , no_rel , 
	):
	#----- determine some arguments and prepare model -----

	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	if models is not None:
		if isinstance(models , tc.nn.Module):
			models = [models]
		for i in range(len(models)):
			models[i] = models[i].eval()

	readable_info = ""
	model_output = []
	dataset_info = []

	#----- gene -----
	#dataset = dataset[:5]
	pbar = tqdm(range(len(dataset)) , ncols = 70)
	generated = ""
	for text_id in pbar:
		#----- get data -----

		data = dataset[text_id:text_id+1]
		sents , ents , anss , data_ent = get_data_from_batch(data)

		if models is not None:
			with tc.no_grad():

				preds = [0 for _ in range(len(models))]
				for i , model in enumerate(models):

					old_device = next(model.parameters()).device
					model = model.cuda()
					preds[i] = model(sents , ents)
					model = model.to(old_device) #如果他本来在cpu上，生成完之后还是把他放回cpu

				#----- get generated output -----

				ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
				generated , pred = generate(preds , data_ent , relations , no_rel , 
						ans_rels = ans_rels , give_me_pred = True)

		#----- form data structure -----
		# text
		text = tokenizer.decode(sents[0][1:-1])

		# entitys
		ents = ents[0]
		for i in range(len(ents)):
			ents[i].append(tokenizer.decode(sents[0][ents[i][0] : ents[i][1]]))
			ents[i][0] = len(tokenizer.decode(sents[0][1:ents[i][0]]))+1 #前方的字符数(+1 is for space)
			ents[i][1] = len(tokenizer.decode(sents[0][1:ents[i][1]]))   #前方的字符数
			ents[i] = [i] + ents[i]
		
		# golden answer
		anss = anss[0]
		for i in range(len(anss)):
			anss[i][2] = relations[anss[i][2]]
		golden_ans = anss

		# model output
		if models is not None:
			got_ans = []
			for x in list(filter(lambda x:x , generated.strip().split("\n"))):
				if x == "":
					continue
				reg = "(.*)\\(.*\\.(\\d*)\\,.*\\.(\\d*)(.*)\\)"
				rel_type , u , v , rev = re.findall(reg , x)[0]
				assert (not rev) or (rev == ",REVERSE")
				if rev: u,v = v,u
				got_ans.append( [int(u)-1,int(v)-1,rel_type] )

		if models is not None:
			for u,v,_ in got_ans:
				model_output.append(
					{
						"doc_id" : text_id+1,
						"ent0_id" : u,
						"ent1_id" : v,
						"list_of_prob" : [float(x) for x in pred[0][u][v]] , 
					}
				)

		dataset_info.append(
			{
				"doc_id" : text_id+1,
				"text" : text,
				"entity_set" : [[int(idx),int(l),int(r),cont] for idx,l,r,cont in ents],
				"list_of_relations" : [[x[0],x[1],relations.index(x[2])] for x in golden_ans] , 
			}
		)

		ents 		= beautiful_str(["id" , "l" , "r" , "content"] , ents)
		golden_ans 	= beautiful_str(["ent0 id" , "ent1 id" , "relation type"] , golden_ans)
		if models is not None:
			got_ans = beautiful_str(["ent0 id" , "ent1 id" , "relation type"] , got_ans)
		else:
			got_ans = "None"

		readable_info += "text-%d:\n%s\n\nentitys:%s\n\ngolden relations:%s\n\nmodel(edge-aware) output:%s\n\n\n" % (
			text_id+1,text , ents , golden_ans , got_ans
		)


		pbar.set_description_str("(Generate)")

	os.makedirs(os.path.dirname(C.gene_file) , exist_ok = True)
	with open(C.gene_file + ".txt" , "w" , encoding = "utf-8") as fil:
		fil.write(readable_info)
	with open(C.gene_file + ".model.json" , "w" , encoding = "utf-8") as fil:
		json.dump(model_output , fil)
	with open(C.gene_file + ".dataset.json" , "w" , encoding = "utf-8") as fil:
		json.dump(dataset_info , fil)


if __name__ == "__main__":
	from config import C, logger

	#----- prepare data and some global variables -----
	data_train , data_test , data_valid , relations, _ = load_data(C , logger)

	if C.watch_type == "train":
		data_watch = data_train
	if C.watch_type == "test":
		data_watch = data_test
	if C.watch_type == "valid":
		data_watch = data_valid

	if C.rel_only: # no no_rel
		no_rel = -1
	else:
		no_rel = len(relations)


	#----- load model -----
	if C.model_save:
		with open(C.model_save , "rb") as fil:
			trained_models = pickle.load(fil)
	else:
		trained_models = None

	#----- generate -----

	generate_output(C , logger , data_watch , trained_models , relations , no_rel)
