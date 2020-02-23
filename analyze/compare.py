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


def compare(
		C , logger , 
		dataset , models_1 , models_2 , generator 
	):
	#----- determine some arguments and prepare model -----

	bert_type = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(bert_type)

	golden = write_keyfile(dataset , generator)

	models_1 = models_1.eval()
	models_2 = models_2.eval()

	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	#----- gene -----

	readable_info = ""
	json_info = []

	all_generated_1 = ""
	all_generated_2 = ""
	for batch_id in tqdm(range(batch_numb) , ncols = 70 , desc = "Generating..."):
		#----- get data -----

		data = dataset[batch_id * batch_size:(batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		with tc.no_grad():

			preds_1 = models_1(sents , ents , output_preds = True)
			preds_2 = models_2(sents , ents , output_preds = True)

			#----- get generated output -----

			ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.gene_in_data else None
			generated_1 = generator(preds_1 , data_ent , ans_rels = ans_rels , split_generate = True)
			generated_2 = generator(preds_2 , data_ent , ans_rels = ans_rels , split_generate = True)

			all_generated_1 += "".join(generated_1)
			all_generated_2 += "".join(generated_2)

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

			# model 1 output
			got_ans_1 = []
			for x in list(filter(lambda x:x , generated_1[text_id].strip().split("\n"))):
				if x == "":
					continue
				reg = "(.*)\\(.*\\.(\\d*)\\,.*\\.(\\d*)(.*)\\)"
				rel_type , u , v , rev = re.findall(reg , x)[0]
				assert (not rev) or (rev == ",REVERSE")
				if rev: u,v = v,u
				got_ans_1.append( [int(u)-1,int(v)-1,rel_type] )

			got_ans_2 = []
			for x in list(filter(lambda x:x , generated_2[text_id].strip().split("\n"))):
				if x == "":
					continue
				reg = "(.*)\\(.*\\.(\\d*)\\,.*\\.(\\d*)(.*)\\)"
				rel_type , u , v , rev = re.findall(reg , x)[0]
				assert (not rev) or (rev == ",REVERSE")
				if rev: u,v = v,u
				got_ans_2.append( [int(u)-1,int(v)-1,rel_type] )


			tmp_ents_s 	= beautiful_str(["id" , "l" , "r" , "content"] , tmp_ents)

			if (not C.gene_in_data) or (not C.gene_no_rel):
				golden_ans_s 	= beautiful_str(["ent 0 id" , "ent 1 id" , "relation type"] , golden_ans)
				got_ans_1_s 	= beautiful_str(["ent 0 id" , "ent 1 id" , "relation type"] , got_ans_1)
				got_ans_2_s 	= beautiful_str(["ent 0 id" , "ent 1 id" , "relation type"] , got_ans_2)

				readable_info += "text-%d:\n%s\n\nentitys:%s\n\ngolden relations:%s\n\nmodel output-1:%s\n\noutput-1:%s\n\n\n" % (
					batch_id*batch_size+text_id+1,text , tmp_ents_s , golden_ans_s , got_ans_1_s , got_ans_2_s
				)

				json_info.append({
					"text-id" : batch_id*batch_size+text_id+1 , 
					"text" 	  : text , 
					"entitys" : intize(tmp_ents , [0,1,2]) , 
					"golden_ans" : intize(golden_ans , [0,1]) , 
					"got_ans_1" : intize(got_ans_1 , [0,1]) , 
					"got_ans_2" : intize(got_ans_2 , [0,1]) , 

				})
			else: #ensure there are exactly the same entity pairs in gold and generated

				try:
					assert [ x[:2] for x in golden_ans ] ==  [ x[:2] for x in got_ans_1 ]
					assert [ x[:2] for x in golden_ans ] ==  [ x[:2] for x in got_ans_2 ]
				except AssertionError:
					pdb.set_trace()
				
				all_ans = []
				for _ins_i in range(len(golden_ans)):
					all_ans.append([
						golden_ans[_ins_i][0] , golden_ans[_ins_i][1] , 
						golden_ans[_ins_i][2] , got_ans_1[_ins_i][2] , got_ans_2[_ins_i][2] , 
					])

				all_ans_s = beautiful_str(["ent 0 id" , "ent 1 id" , "golden" , "model 1" , "model 2"] , all_ans)
				readable_info += "text-%d:\n%s\n\nentitys:%s\n\noutputs:%s\n\n\n" % (
					text_id+1 , text , tmp_ents_s , all_ans_s , 
				)

				json_info.append({
					"text-id" 	: batch_id*batch_size+text_id+1 , 
					"text" 	  	: text , 
					"entitys" 	: intize(tmp_ents , [0,1,2]) , 
					"relations" : intize(all_ans , [0,1]) , 
				})



	os.makedirs(os.path.dirname(C.gene_file) , exist_ok = True)
	with open(C.gene_file + ".txt" , "w" , encoding = "utf-8") as fil:
		fil.write(readable_info)
	with open(C.gene_file + ".json" , "w" , encoding = "utf-8") as fil:
		json.dump(json_info , fil)
	print ("score (model 1): %.4f %.4f" % get_f1(golden , all_generated_1 , 
			is_file_content = True , no_rel_name = generator.get_no_rel_name()) )
	print ("score (model 2): %.4f %.4f" % get_f1(golden , all_generated_2 , 
			is_file_content = True , no_rel_name = generator.get_no_rel_name()) )

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

	if not C.model_save or not C.model_save_2:
		assert "model information incomplete"

	with open(C.model_save , "rb") as fil:
		models_1 = pickle.load(fil)
	with open(C.model_save_2 , "rb") as fil:
		models_2 = pickle.load(fil)

	models_1 = EnsembleModel(models_1)
	models_2 = EnsembleModel(models_2)

	#----- generate -----

	compare(C , logger , data_watch , models_1 , models_2 , generator)
