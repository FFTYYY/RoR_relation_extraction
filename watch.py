'''
	watch trained model results
'''
from transformers import BertModel , BertTokenizer
from dataloader import get_dataloader
from tqdm import tqdm
import torch as tc
import pdb
import os , sys
import math
from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup
from generate import Generator
from test import test
from utils.train_util import pad_sents , get_data_from_batch
from utils.scorer import get_f1
import fitlog
import pickle
import re
from YTools.universe.beautiful_str import beautiful_str
import json
from config import get_config

from main import load_data , initialize

#fitlog.debug()
#fitlog.commit(__file__)


def generate_output(
		C , logger , 
		dataset , models , generator 
	):
	#----- determine some arguments and prepare model -----

	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	if models is not None:
		if isinstance(models , tc.nn.Module):
			models = [models]
		for i in range(len(models)):
			models[i] = models[i].eval()
	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	device = tc.device(C.device)

	readable_info = ""
	model_output = []
	dataset_info = []
	all_generated = ""

	#----- gene -----

	#dataset = dataset[:5]
	pbar = tqdm(range(batch_numb) , ncols = 70)

	generated = ""
	for batch_id in pbar:
		#----- get data -----

		data = dataset[batch_id * batch_size:(batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		if models is not None:
			with tc.no_grad():

				preds = [0 for _ in range(len(models))]
				for i , model in enumerate(models):

					old_device = next(model.parameters()).device
					model = model.to(device)
					preds[i] = model(sents , ents)
					model = model.to(old_device) #如果他本来在cpu上，生成完之后还是把他放回cpu

				#----- get generated output -----

				ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.gene_in_data else None
				generated , pred = generator(preds , data_ent , 
						ans_rels = ans_rels , give_me_pred = True , split_generate = True)
				all_generated += "".join(generated)

		for text_id in range(len(data)):
			tmp_pred = pred[text_id]

			#----- form data structure -----
			# text
			tmp_sents = sents[text_id]
			while tmp_sents[-1] == 0: # remove padding
				tmp_sents = tmp_sents[:-1]
			text = tokenizer.decode(tmp_sents[1:-1])

			# entitys
			tmp_ents = ents[text_id]
			for i in range(len(tmp_ents)):
				tmp_ents[i].append(tokenizer.decode(tmp_sents[tmp_ents[i][0] : tmp_ents[i][1]]))
				tmp_ents[i][0] = len(tokenizer.decode(tmp_sents[1:tmp_ents[i][0]]))+1 #前方的字符数(+1 is for space)
				tmp_ents[i][1] = len(tokenizer.decode(tmp_sents[1:tmp_ents[i][1]]))   #前方的字符数
				tmp_ents[i] = [i] + tmp_ents[i]
			
			# golden answer
			tmp_anss = anss[text_id]
			for i in range(len(tmp_anss)):
				tmp_anss[i][2] = relations[tmp_anss[i][2]]
			golden_ans = tmp_anss

			# model output
			if models is not None:
				got_ans = []
				for x in list(filter(lambda x:x , generated[text_id].strip().split("\n"))):
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
							"list_of_prob" : [float(x) for x in tmp_pred[u][v]] , 
						}
					)

			dataset_info.append(
				{
					"doc_id" : text_id+1,
					"text" : text,
					"entity_set" : [[int(idx),int(l),int(r),cont] for idx,l,r,cont in tmp_ents],
					"list_of_relations" : [[x[0],x[1],relations.index(x[2])] for x in golden_ans] , 
				}
			)

			tmp_ents 		= beautiful_str(["id" , "l" , "r" , "content"] , tmp_ents)
			golden_ans 	= beautiful_str(["ent0 id" , "ent1 id" , "relation type"] , golden_ans)
			if models is not None:
				got_ans = beautiful_str(["ent0 id" , "ent1 id" , "relation type"] , got_ans)
			else:
				got_ans = "None"

			readable_info += "text-%d:\n%s\n\nentitys:%s\n\ngolden relations:%s\n\nmodel(edge-aware) output:%s\n\n\n" % (
				text_id+1,text , tmp_ents , golden_ans , got_ans
			)


		pbar.set_description_str("(Generate)")

	os.makedirs(os.path.dirname(C.gene_file) , exist_ok = True)
	with open(C.gene_file + ".txt" , "w" , encoding = "utf-8") as fil:
		fil.write(readable_info)
	with open(C.gene_file + ".generate.txt" , "w" , encoding = "utf-8") as fil:
		fil.write(all_generated)
	with open(C.gene_file + ".model.json" , "w" , encoding = "utf-8") as fil:
		json.dump(model_output , fil)
	with open(C.gene_file + ".dataset.json" , "w" , encoding = "utf-8") as fil:
		json.dump(dataset_info , fil)


if __name__ == "__main__":
	C , logger = get_config()
	fitlog.debug()
	
	#----- prepare data and some global variables -----
	data_train , data_test , data_valid , relations, rel_weights = load_data(C , logger)
	_ , loss_func , generator = initialize(C , logger , relations , rel_weights)

	if C.watch_type == "train":
		data_watch = data_train
	if C.watch_type == "test":
		data_watch = data_test
	if C.watch_type == "valid":
		data_watch = data_valid

	#----- load model -----
	if C.model_save:
		with open(C.model_save , "rb") as fil:
			trained_models = pickle.load(fil)
	else:
		trained_models = None

	#----- test -----
	if trained_models is not None:

		micro_f1 , macro_f1 , _ = test(
			C , logger , 
			data_watch , trained_models , 
			loss_func , generator , 
			mode = "test" , epoch_id = 0 , run_name = "0" , need_generated = False , 
		)
		print ("test result: %.2f %.2f" % (micro_f1 , macro_f1))

	#----- generate -----

	generate_output(C , logger , data_watch , trained_models , generator)
