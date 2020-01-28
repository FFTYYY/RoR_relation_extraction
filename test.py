from tqdm import tqdm
import torch as tc
import pdb
import os , sys
import math
import fitlog
import re
from utils.scorer import get_f1
from utils.train_util import pad_sents , get_data_from_batch
from utils.write_keyfile import write_keyfile

def before_test(C , logger , dataset , models):
	
	if isinstance(models , tc.nn.Module):
		models = [models]
	for i in range(len(models)):
		models[i] = models[i].eval()

	device = tc.device(C.device)
	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	return device , batch_size , batch_numb , models

def get_output(C , logger , 
		models , device , loss_func , generator , 
		sents , ents , anss , data_ent , 
	):


	preds = [0 for _ in range(len(models))]
	for i , model in enumerate(models):

		old_device = next(model.parameters()).device
		model = model.to(device)
		preds[i] = model(sents , ents)
		model = model.to(old_device) #如果他本来在cpu上，生成完之后还是把他放回cpu

		loss = loss_func(preds[i] , anss , ents)

	ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
	generated = generator(preds , data_ent , ans_rels = ans_rels)

	#pred_map = pred.max(-1)[1] #(ne , ne)

	return model , preds , loss , generated

def get_evaluate(C , logger , mode , generated , generator , test_data = None):

	golden = write_keyfile(test_data , generator)

	micro_f1 , macro_f1 = get_f1(golden , generated , is_file_content = True , no_rel = generator.get_no_rel_name())
	micro_f1 , macro_f1 = micro_f1 * 100 , macro_f1 * 100

	return micro_f1 , macro_f1


def test(C , logger , 
		dataset , models , 
		loss_func , generator , 
		mode = "valid" , epoch_id = 0 , run_name = "0" , need_generated = False , 
	):
		
	device , batch_size , batch_numb , models = before_test(C , logger , dataset , models)

	pbar = tqdm(range(batch_numb) , ncols = 70)
	avg_loss = 0
	generated = ""
	for batch_id in pbar:


		data = dataset[batch_id * batch_size : (batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		with tc.no_grad():
			model , preds , loss , partial_generated = get_output(
				C,logger,models,device,loss_func,generator,sents,ents,anss,data_ent
			)
		generated += partial_generated
		avg_loss += float(loss) / len(models)

		pbar.set_description_str("(Test )Epoch {0}".format(epoch_id))
		pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))

	micro_f1 , macro_f1 = get_evaluate(C , logger , mode , generated , generator , dataset)

	#print (result)
	logger.log ("-----Epoch {} tested. Micro F1 = {:.2f}% , Macro F1 = {:.2f}% , loss = {:.4f}".
			format(epoch_id , micro_f1, macro_f1, avg_loss / batch_numb))
	logger.log("\n")

	fitlog.add_metric(micro_f1 , step = epoch_id , name = "({0})micro f1".format(run_name)) 
	fitlog.add_metric(macro_f1 , step = epoch_id , name = "({0})macro f1".format(run_name)) 

	if need_generated:
		return micro_f1 , macro_f1 , avg_loss , generated

	return micro_f1 , macro_f1 , avg_loss