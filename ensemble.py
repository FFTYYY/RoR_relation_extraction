import torch as tc
import torch.nn as nn
import pdb
from tqdm import tqdm
import os , sys
from utils.train_util import pad_sents , pad_ents , pad_anss
from config import C , logger
from dataloader import run as read_data , relations , id2rel
from models.gene_func import generate_from_pred

def ensemble_generate(relation_typs , no_rel , preds , data_ent , rel_id2name , fil):
		
		bs , ne , _ , d = preds[0].size()

		pred = 0
		for k in range(len(preds)):
			preds[k] = tc.softmax(preds[k] , dim = -1)
			pred += preds[k]
		pred /= len(preds)

		generate_from_pred(relation_typs , no_rel , pred , data_ent , rel_id2name , fil)



def ensemble_test(relation_typs , no_rel , dataset , models):
	for model in models:
		model.eval()
	batch_size = 8

	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)
	pbar = tqdm(range(batch_numb) , ncols = 70)
	res_file = open(C.tmp_file_name , "w" , encoding = "utf-8")

	for batch_id in pbar:
		data = dataset[batch_id * batch_size : (batch_id+1) * batch_size]

		sents = [x.abs  for x in data] 
		ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
		anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]
		data_ent = [x.ents for x in data] 

		sents 	= pad_sents(sents)
		sents = tc.LongTensor(sents).cuda()

		with tc.no_grad():
			preds = [0 for _ in range(len(models))]
			for i , model in enumerate(models):
				model = model.cuda()
				preds[i] = model(sents , ents)
				model = model.cpu()

			if C.rel_only:
				ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss]
			else:
				ans_rels = None

			ensemble_generate(relation_typs , no_rel , preds , data_ent , id2rel , res_file , ans_rels = ans_rels)

		pbar.set_description_str("Final Test")

	res_file.close()
	os.system("perl {script} {result_file} {key_file} > {result_save}".format(
		script 		= C.test_script ,
		result_file = C.tmp_file_name,
		key_file 	= C.test_rels ,
		result_save = C.tmp_file_name + ".imm"
	))
	with open(C.tmp_file_name + ".imm" , "r" , encoding = "utf-8") as rfil:
		result = rfil.read()
	logger.log (result)
	logger.log ("Ensemble tested.")

	model = model.train()
	#pdb.set_trace()


