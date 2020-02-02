from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup
from tqdm import tqdm
import torch as tc
import pdb
import os , sys
import math
import fitlog
import pickle
from models import get_model
from test import test
from utils.train_util import get_data_from_batch

#fitlog.commit(__file__)


def before_train(C , logger , train_data , n_rel_typs):

	batch_numb = (len(train_data) // C.batch_size) + int((len(train_data) % C.batch_size) != 0)
	device = tc.device(C.device)

	model = get_model()(
		n_rel_typs = n_rel_typs , dropout = C.dropout , 
		device = device , 
		gnn = C.gnn , matrix_trans = C.matrix_trans , matrix_nlayer = C.matrix_nlayer , 
	).to(device)

	optimizer = tc.optim.Adam(params = model.parameters() , lr = C.lr)

	scheduler_makers = {
		"linear": get_linear_schedule_with_warmup ,
		"cosine": get_cosine_schedule_with_warmup ,
	}
	scheduler = scheduler_makers[C.scheduler](
		optimizer = optimizer , 
		num_warmup_steps = int(C.warmup_prop * batch_numb * C.epoch_numb), 
		num_training_steps = batch_numb * C.epoch_numb , 
	)

	return (batch_numb , device) , (model , optimizer , scheduler)

def update_batch(C , logger , 
		model , optimizer , scheduler , loss_func ,  
		sents , ents , anss , data_ent , 
	):
	pred = model(sents , ents)
	loss = loss_func(pred , anss , ents)

	#----- backward -----
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()

	return loss , pred


def train(C , logger , train_data , valid_data , loss_func , generator , n_rel_typs , run_name = "0" , test_data = None):	
	(batch_numb , device) , (model , optimizer , scheduler) = before_train(
		C , logger , train_data , n_rel_typs
	)
	#----- iterate each epoch -----

	best_epoch = -1
	best_metric = -1
	for epoch_id in range(C.epoch_numb):

		pbar = tqdm(range(batch_numb) , ncols = 70)
		avg_loss = 0
		for batch_id in pbar:
			#----- get data -----
			data = train_data[batch_id * C.batch_size : (batch_id+1) * C.batch_size]
			sents , ents , anss , data_ent = get_data_from_batch(data , device = device)

			loss , pred = update_batch(
				C , logger , model , optimizer , scheduler , loss_func , sents , ents , anss , data_ent
			)

			avg_loss += float(loss)
			fitlog.add_loss(value = float(loss) , step = epoch_id * batch_numb + batch_id , 
					name = "({0})train loss".format(run_name))

			pbar.set_description_str("(Train)Epoch %d" % (epoch_id))
			pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))
		logger.log ("Epoch %d ended. avg_loss = %.4f" % (epoch_id , avg_loss / batch_numb))


		micro_f1 , macro_f1 , test_loss = test(
			C , logger , 
			valid_data , model , 
			loss_func , generator , 
			"valid" , epoch_id   , run_name , 
		)

		if C.valid_metric in ["macro*micro" , "micro*macro"]:
			metric = macro_f1 * micro_f1
		elif C.valid_metric == "macro":
			metric = macro_f1
		elif C.valid_metric == "micro":
			metric = micro_f1
		else:
			assert False

		if best_metric < metric:
			best_epoch = epoch_id
			best_metric = metric
			with open(C.tmp_file_name + ".model" + "." + str(run_name) , "wb") as fil:
				pickle.dump(model , fil)
			
		#	fitlog.add_best_metric(best_macro_f1 , name = "({0})macro f1".format(ensemble_id))

		model = model.train()

	if not C.no_valid: #reload best model
		with open(C.tmp_file_name + ".model" + "." + str(run_name) , "rb") as fil:
			model = pickle.load(fil) #load best valid model
		logger.log("reloaded best model at epoch %d" % best_epoch)

	if test_data is not None:
		final_micro_f1 , final_macro_f1 , final_test_loss = test(
			C , logger , 
			test_data , model , 
			loss_func , generator , 
			"test" , epoch_id   , run_name , 
		)

	return model , best_metric

