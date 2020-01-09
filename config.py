import argparse
import os
from utils.logger import Logger
from pprint import pformat
import random
import torch as tc
import numpy as np
from utils.tmp_file import random_tmp_name
import fitlog

fitlog.commit(__file__)
fitlog.set_log_dir("logs")

_par = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#dataloader
_par.add_argument("--train_text_1" 	, type = str , default = "./data/semeval_2018_task7/1.1.text.xml")
_par.add_argument("--train_rels_1" 	, type = str , default = "./data/semeval_2018_task7/1.1.relations.txt")
_par.add_argument("--train_text_2" 	, type = str , default = "./data/semeval_2018_task7/1.2.text.xml")
_par.add_argument("--train_rels_2" 	, type = str , default = "./data/semeval_2018_task7/1.2.relations.txt")
_par.add_argument("--test_text" 	, type = str , default = "./data/semeval_2018_task7/2.test.text.xml")
_par.add_argument("--test_rels" 	, type = str , default = "./data/semeval_2018_task7/keys.test.2.txt")
_par.add_argument("--test_script" 	, type = str , default = "./data/semeval_2018_task7/semeval2018_task7_scorer-v1.2.pl")
_par.add_argument("--dataset", default = 'semeval_2018_task7')

_par.add_argument("--rel_only" 				, action = "store_true" , default = False)
_par.add_argument("--no_rel_weight" 		, type = float , default = 0.05)
_par.add_argument("--rel_weight_smooth" 	, type = float , default = 0)
_par.add_argument("--rel_weight_norm" 		, action = 'store_true')

#model selection
_par.add_argument("--model" 		, type = str , default = "naive_bert" , 
	choices = [
		"naive_bert" , 
		"graph_trans" , 
])

#model structure 

_par.add_argument("--dropout" 		, type = float , default = 0.0)
_par.add_argument("--loss" 			, type = str , default = "loss_3")

#training arguments
_par.add_argument("--batch_size" 	, type = int , default = 8)
_par.add_argument("--epoch_numb" 	, type = int , default = 50)
_par.add_argument("--lr" 			, type = float , default = 1e-4)
_par.add_argument("--fine_tune" 	, action = "store_true" , default = False)
_par.add_argument("--n_warmup"		, type = int , default = 400)


_par.add_argument("--test_mode" 	, action = "store_true" , default = False)
_par.add_argument("--ensemble_size" , type = int , default = 5)

#others 
_par.add_argument("--gpus" 			, type = str	, default = "0")
_par.add_argument("--seed" 			, type = int , default = 2333)
_par.add_argument("--log_file" 		, type = str , default = "log.txt")
_par.add_argument("--no_log" 		, action = "store_true" , default = False)

#---------------------------------------------------------------------------------------------------

C = _par.parse_args()

#----- do some check -----

for file in [C.train_text_1 , C.train_rels_1 , C.train_text_2 , C.train_rels_2 , C.test_text , C.test_rels ]:
	if C.dataset not in file:
		print('[Warn] Dataset and training files do not match.')
		import pdb;pdb.set_trace()

#----- make logger -----

if C.test_mode:
	C.log_file += ".test"

logger = Logger(C.log_file)
logger.log = logger.log_print_w_time
if C.no_log:
	logger.log = logger.nolog

C.tmp_file_name = random_tmp_name()

#----- other stuff -----
fitlog.add_hyper(C)

logger.log ("------------------------------------------------------")
logger.log (pformat(C.__dict__))
logger.log ("------------------------------------------------------")

#fitlog不支持list，所以把字符串作为参数，之后再转成list
C.gpus = [int(x) for x in C.gpus.strip().split(",")] 

#----- initialize -----

if C.seed > 0:
	random.seed(C.seed)
	tc.manual_seed(C.seed)
	np.random.seed(C.seed)
	tc.cuda.manual_seed_all(C.seed)
	tc.backends.cudnn.deterministic = True
	tc.backends.cudnn.benchmark = False

	logger.log ("Seed set. %d" % (C.seed))

tc.cuda.set_device(C.gpus[0])
