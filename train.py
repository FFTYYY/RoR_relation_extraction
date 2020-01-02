from config import C , logger
from dataloader import read_data
from tqdm import tqdm
from utils.train_util import pad_sents , pad_ents , pad_anss
import torch as tc
from models import models
import pdb
import os , sys
import math
from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup
from models.loss_func import loss_funcs
from models.gene_func import generate
from ensemble import ensemble_test
from utils.scorer import get_f1

def load_data():
	data_train , data_test, relations, rel_weights = read_data(
		C.train_text_1 , C.train_rels_1 ,
		C.train_text_2 , C.train_rels_2 ,
		C.test_text , C.test_rels , 
	)

	return data_train , data_test, relations, rel_weights

def valid(relations, rel_weights, relation_typs , no_rel , dataset , model , epoch_id = 0):
	def id2rel(i):
		return relations[i]

	model = model.eval()
	batch_size = 8

	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)
	pbar = tqdm(range(batch_numb) , ncols = 70)
	avg_loss = 0
	res_file = open(C.tmp_file_name , "w" , encoding = "utf-8")
	loss_func = loss_funcs[C.loss]

	for batch_id in pbar:
		data = dataset[batch_id * batch_size : (batch_id+1) * batch_size]

		sents = [x.abs  for x in data] 
		ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
		anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]
		data_ent = [x.ents for x in data] 

		sents 	= pad_sents(sents)
		sents = tc.LongTensor(sents).cuda()

		with tc.no_grad():
			pred = model(sents , ents)
			loss = loss_func(relation_typs , no_rel , pred , anss , ents, rel_weights)
			#loss = 0.
			if C.rel_only:
				ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss]
			else:
				ans_rels = None
			generate(relation_typs , no_rel , pred , data_ent , id2rel , res_file , ans_rels = ans_rels)

		try:
			assert not math.isnan(float(loss))
		except AssertionError:
			pdb.set_trace()

		avg_loss += float(loss)

		pbar.set_description_str("(Test )Epoch %d" % (epoch_id + 1))
		pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))

	res_file.close()
	if C.dataset == 'semeval_2018_task7':
		os.system("perl {script} {result_file} {key_file} > {result_save}".format(
			script 		= C.test_script ,
			result_file = C.tmp_file_name,
			key_file 	= C.test_rels ,
			result_save = C.tmp_file_name + ".imm"
		))
		with open(C.tmp_file_name + ".imm" , "r" , encoding = "utf-8") as rfil:
			result = rfil.read()
		logger.log (result)
		logger.log ("Epoch {} tested. loss={:.4f}".format(epoch_id + 1 , avg_loss / batch_numb))
	else:
		f1_micro, f1_macro = get_f1([C.test_rels, C.train_rels_1, C.train_rels_2], C.test_rels, C.tmp_file_name)
		logger.log ("Epoch {} tested. F1_mi={:.2f}%, F1_ma={:.2f}%, loss={:.4f}".format(epoch_id + 1 , f1_micro * 100, f1_macro * 100, avg_loss / batch_numb))

	model = model.train()
	#pdb.set_trace()

def train(train_data , test_data, relations, rel_weights):

	if C.rel_only:
		relation_typs , no_rel = len(relations) , -1
	else:
		relation_typs , no_rel = len(relations) + 1 , len(relations)
		rel_weights += [C.no_rel_weight]

	model = models[C.model](relation_typs = relation_typs , dropout = C.dropout).cuda()

	batch_numb = (len(train_data) // C.batch_size) + int((len(train_data) % C.batch_size) != 0)

	optimizer = tc.optim.Adam(params = model.parameters() , lr = C.lr)
	scheduler = get_cosine_schedule_with_warmup(
		optimizer = optimizer , 
		num_warmup_steps = C.n_warmup , 
		num_training_steps = batch_numb * C.epoch_numb , 
	)
	loss_func = loss_funcs[C.loss]

	for epoch_id in range(C.epoch_numb):

		pbar = tqdm(range(batch_numb) , ncols = 70)
		avg_loss = 0
		for batch_id in pbar:
			data = train_data[batch_id * C.batch_size : (batch_id+1) * C.batch_size]

			sents = [x.abs  for x in data] 
			ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
			anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]

			sents = pad_sents(sents)
			sents = tc.LongTensor(sents).cuda()

			pred = model(sents , ents)
			loss = loss_func(relation_typs , no_rel , pred , anss , ents, class_weight=rel_weights)

			try:
				assert loss.item() == loss.item()
			except AssertionError:
				pdb.set_trace()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			avg_loss += float(loss)

			pbar.set_description_str("(Train)Epoch %d" % (epoch_id + 1))
			pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))
		logger.log ("Epoch %d ended. avg_loss = %.4f" % (epoch_id + 1 , avg_loss / batch_numb))
		valid(relations, rel_weights, relation_typs , no_rel , test_data , model , epoch_id)

	return model

if __name__ == "__main__":

	data_train , data_test, relations, rel_weights = load_data()
	def id2rel(i):
		return relations[i]

	trained_models = []

	for i in range(C.ensemble_size):
		model = train(data_train , data_test, relations, rel_weights)
		model = model.cpu()
		trained_models.append(model)

	if C.rel_only:
		relation_typs , no_rel = len(relations) , -1
	else:
		relation_typs , no_rel = len(relations) + 1 , len(relations)
		rel_weights += [C.no_rel_weight]
	ensemble_test(relation_typs , no_rel , data_test , trained_models, id2rel)

	os.system("rm %s" % C.tmp_file_name)
