from train import train
from dataloader import get_dataloader
from loss import get_loss_func
from generate import Generator
from test import test
import fitlog
import pickle 
import pdb

def load_data(C , logger):
	data_train , data_test , data_valid , relations, rel_weights = get_dataloader(C.dataset)(
		C , logger , 
		C.train_text_1 , C.train_rels_1 ,
		C.train_text_2 , C.train_rels_2 ,
		C.test_text  , C.test_rels ,
		C.valid_text , C.valid_rels ,
		C.dataset, C.rel_weight_smooth, C.rel_weight_norm,
	)

	logger.log("num of sents / entity pairs in train: %d / %d" % ( len(data_train) , sum([ len(x.ents) * (len(x.ents)-1) / 2 for x in data_train  ])  ))
	logger.log("num of sents / instances in test    : %d / %d" % ( len(data_test ) , sum([ len(x.ans) for x in data_test  ])  ))
	logger.log("num of sents / instances in valid   : %d / %d" % ( len(data_valid) , sum([ len(x.ans) for x in data_valid  ])  ))

	return data_train , data_test , data_valid , relations, rel_weights

def initialize(C , logger , relations , rel_weights):
	
	n_rel_typs = len(relations)

	no_rel = 0
	no_rel = relations.index(C.no_rel_name)

	rel_weights[no_rel] = C.no_rel_weight
	logger.log("relations : {0}".format(relations))
	logger.log("rel_weights : {0}".format(rel_weights))

	#assert len(rel_weights) == 7

	loss_func = get_loss_func(C.loss , no_rel = no_rel , class_weight = rel_weights)
	generator = Generator(C , relations = relations , no_rel = no_rel)

	#pdb.set_trace()

	return n_rel_typs , loss_func , generator


def main():
	from config import get_config

	C , logger = get_config()

	#----- prepare data and some global variables -----
	data_train , data_test , data_valid , relations, rel_weights = load_data(C , logger)

	n_rel_typs , loss_func , generator = initialize(C , logger , relations, rel_weights)
	#----- train & test -----
	trained_models = []
	for i in range(C.ensemble_size):
		model = train(
			C , logger , 
			data_train , data_valid , 
			loss_func , generator , n_rel_typs  , 
			run_name = str(i) , test_data = data_test , 
		)
		model = model.cpu()
		trained_models.append(model)

	#----- ensemble test -----
	micro_f1 , macro_f1 , loss = test(
		C , logger , 
		data_test , trained_models , 
		loss_func , generator , 
		mode = "test" , epoch_id = C.epoch_numb , run_name = 'final', 
	)
	fitlog.add_hyper(macro_f1 , name = "result")

	#----- save ensembled model -----
	if C.model_save:
		with open(C.model_save , "wb") as fil:
			pickle.dump(trained_models , fil)
	logger.log("final model saved at %s" % C.model_save)

	#----- finish -----
	fitlog.finish()

if __name__ == "__main__":
	main()
