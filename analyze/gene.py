'''
	比较两个模型
'''
import torch as tc
import pdb
import os , sys
import fitlog
import re
import pickle
from transformers import BertTokenizer
from tqdm import tqdm
from utils.train_util import get_data_from_batch
from utils.scorer import get_f1
from utils.composed_model import TwoPhaseModel , EnsembleModel
from utils.write_keyfile import write_keyfile
from utils.others import intize
from YTools.universe.beautiful_str import beautiful_str
from config import get_config
import json

from main import load_data , initialize

#fitlog.debug()
#fitlog.commit(__file__)


def gene( C , logger , dataset , models , generator ):
	#----- determine some arguments and prepare model -----

	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	golden = write_keyfile(dataset , generator)

	models = models.eval()

	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	#----- gene -----

	readable_info = ""
	json_info = []

	all_generated = ""
	for batch_id in tqdm(range(batch_numb) , ncols = 70 , desc = "Generating..."):
		#----- get data -----

		data = dataset[batch_id * batch_size:(batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		with tc.no_grad():

			preds = models(sents , ents , output_preds = True)

			#----- get generated output -----

			ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.gene_in_data else None
			generated = generator(preds , data_ent , ans_rels = ans_rels , split_generate = True)

			all_generated += "".join(generated)

		for text_id in range(len(data)):

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
			
			# model 1 output
			got_ans = []
			for x in list(filter(lambda x:x , generated[text_id].strip().split("\n"))):
				if x == "":
					continue
				reg = "(.*)\\(.*\\.(\\d*)\\,.*\\.(\\d*)(.*)\\)"
				rel_type , u , v , rev = re.findall(reg , x)[0]
				assert (not rev) or (rev == ",REVERSE")
				if rev: u,v = v,u
				got_ans.append( [int(u)-1,int(v)-1,rel_type] )


			tmp_ents_s 	= beautiful_str(["id" , "l" , "r" , "content"] , tmp_ents)

			got_ans_s = beautiful_str(["ent 0 id" , "ent 1 id" , "relation"] , got_ans)

			readable_info += "text-%d:\n%s\n\nentitys:%s\n\noutputs:%s\n\n\n" % (
				batch_id*batch_size+text_id+1 , text , tmp_ents_s , got_ans_s , 
			)

			json_info.append({
				"text-id" 	: batch_id*batch_size+text_id+1 , 
				"text" 	  	: text , 
				"entitys" 	: intize(tmp_ents , [0,1,2]) , 
				"relations" : intize(got_ans , [0,1]) , 
			})



	os.makedirs(os.path.dirname(C.gene_file) , exist_ok = True)
	with open(C.gene_file + ".txt" , "w" , encoding = "utf-8") as fil:
		fil.write(readable_info)
	with open(C.gene_file + ".json" , "w" , encoding = "utf-8") as fil:
		json.dump(json_info , fil)
	print ("score : %.4f %.4f" % get_f1(golden , all_generated , 
			is_file_content = True , no_rel_name = generator.get_no_rel_name()) )

def gene_golden(C , logger , dataset , generator ):
	#----- determine some arguments and prepare model -----

	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	#----- gene -----

	readable_info = ""
	json_info = []

	for batch_id in tqdm(range(batch_numb) , ncols = 70 , desc = "Generating..."):
		#----- get data -----

		data = dataset[batch_id * batch_size:(batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))


		for text_id in range(len(data)):

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


			tmp_ents_s 	= beautiful_str(["id" , "l" , "r" , "content"] , tmp_ents)
			golden_ans_s = beautiful_str(["ent 0 id" , "ent 1 id" , "relation"] , golden_ans)

			readable_info += "text-%d:\n%s\n\nentitys:%s\n\noutputs:%s\n\n\n" % (
				batch_id*batch_size+text_id+1 , text , tmp_ents_s , golden_ans_s , 
			)

			json_info.append({
				"text-id" 	: batch_id*batch_size+text_id+1 , 
				"text" 	  	: text , 
				"entitys" 	: intize(tmp_ents , [0,1,2]) , 
				"relations" : intize(golden_ans , [0,1]) , 
			})



	os.makedirs(os.path.dirname(C.gene_file) , exist_ok = True)
	with open(C.gene_file + ".txt" , "w" , encoding = "utf-8") as fil:
		fil.write(readable_info)
	with open(C.gene_file + ".json" , "w" , encoding = "utf-8") as fil:
		json.dump(json_info , fil)


if __name__ == "__main__":
	C , logger = get_config()
	#fitlog.debug()
	C.info += "-watch"
	
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

	if C.model_save == "golden":
		gene_golden(C , logger , data_watch , generator)
	else:
		with open(C.model_save , "rb") as fil:
			models = pickle.load(fil)

		models = EnsembleModel(models)

		gene(C , logger , data_watch , models , generator)
