from dataloader import read_data
from tqdm import tqdm
import torch as tc
import pdb
import os , sys
import math
from loss import loss_funcs
from generate import generate
from utils.scorer import get_f1
from utils.train_util import pad_sents , get_data_from_batch
import fitlog
import re

fitlog.commit(__file__)

def test(
		C , logger , 
		dataset , models , 
		relations , rel_weights , no_rel , 
		mode = "valid" , epoch_id = 0 , ensemble_id = 0 , 
	):
	#----- determine some arguments and prepare model -----

	if isinstance(models , tc.nn.Module):
		models = [models]
	for i in range(len(models)):
		models[i] = models[i].eval()

	device = tc.device(C.device)
	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)
	loss_func = loss_funcs[C.loss]

	#----- test -----

	pbar = tqdm(range(batch_numb) , ncols = 70)
	avg_loss = 0
	generated = ""
	for batch_id in pbar:

		#----- get data -----

		data = dataset[batch_id * batch_size : (batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		with tc.no_grad():

		#----- get output & loss for each model -----
			preds = [0 for _ in range(len(models))]
			for i , model in enumerate(models):

				old_device = next(model.parameters()).device
				model = model.to(device)
				preds[i] = model(sents , ents)
				model = model.to(old_device) #如果他本来在cpu上，生成完之后还是把他放回cpu

				loss = loss_func(preds[i] , anss , ents , no_rel = no_rel , class_weight = rel_weights)
				avg_loss += float(loss) / len(models)

		#----- get generated output -----

			ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
			generated += generate(preds , data_ent , relations , no_rel , ans_rels = ans_rels)

		
		pbar.set_description_str("(Test )Epoch {0}".format(epoch_id))
		pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))

	#----- evaluate from generated -----

	if C.dataset == 'semeval_2018_task7':
		with open(C.tmp_file_name , "w" , encoding = "utf-8") as ofil:
			ofil.write(generated)

		key_file = C.valid_rels if mode == "valid" else C.test_rels

		os.system("perl {script} {output_file} {key_file} > {result_file}".format(
			script 		= C.test_script ,
			output_file = C.tmp_file_name,
			key_file 	= key_file ,
			result_file = C.tmp_file_name + ".imm"
		))
		with open(C.tmp_file_name + ".imm" , "r" , encoding = "utf-8") as rfil:
			result = rfil.read()

		#pdb.set_trace()

		if not result.strip(): #submission is empty
			micro_f1 = 0
			macro_f1 = 0
		else:
			micro_f1 = float(re.findall('Micro-averaged result[\\s\\S]*?F1 = *(\\d*?\\.\\d*?)%', result)[0])
			macro_f1 = float(re.findall('Macro-averaged result[\\s\\S]*?F1 = *(\\d*?\\.\\d*?)%', result)[0])

		os.system("rm %s" % C.tmp_file_name)
		os.system("rm %s.imm" % C.tmp_file_name)
	else:
		micro_f1 , macro_f1 = get_f1(C.test_rels, C.tmp_file_name)
		micro_f1 , macro_f1 = micro_f1 * 100 , macro_f1 * 100

	#----- record the results -----
	#print (result)
	logger.log ("-----Epoch {} tested. Micro F1 = {:.2f}% , Macro F1 = {:.2f}% , loss = {:.4f}".
			format(epoch_id , micro_f1, macro_f1, avg_loss / batch_numb))
	logger.log("\n")

	fitlog.add_metric(micro_f1 , step = epoch_id , name = "({0})micro f1".format(ensemble_id)) 
	fitlog.add_metric(macro_f1 , step = epoch_id , name = "({0})macro f1".format(ensemble_id)) 

	return micro_f1 , macro_f1