import argparse
import os
from utils.logger import Logger
from pprint import pformat
import random
import torch as tc
import numpy as np
from utils.tmp_file import random_tmp_name
import fitlog

def before_parse_t2g(par):

	#---------------------------------------------------------------------------------------------------

	#dataloader
	par.add_argument("--train_text_1" 	, type = str , default = "./data/semeval_2018_task7/1.1.text.xml")
	par.add_argument("--train_rels_1" 	, type = str , default = "./data/semeval_2018_task7/1.1.relations.txt")
	par.add_argument("--train_text_2" 	, type = str , default = "./data/semeval_2018_task7/1.2.text.xml")
	par.add_argument("--train_rels_2" 	, type = str , default = "./data/semeval_2018_task7/1.2.relations.txt")

	par.add_argument("--valid_text" , type = str , default = "./data/semeval_2018_task7/1.1.test.text.xml")
	par.add_argument("--valid_rels" , type = str , default = "./data/semeval_2018_task7/keys.test.1.1.txt")

	par.add_argument("--test_text" 	, type = str , default = "./data/semeval_2018_task7/2.test.text.xml")
	par.add_argument("--test_rels" 	, type = str , default = "./data/semeval_2018_task7/keys.test.2.txt")

	par.add_argument("--test_script", type = str , default = "./data/semeval_2018_task7/semeval2018_task7_scorer-v1.2.pl")
	par.add_argument("--dataset" 	, default = 'semeval_2018_task7')

	#model selection
	par.add_argument("--gnn"      	  , action = "store_true" , default = False)
	par.add_argument("--matrix_trans" , action = "store_true" , default = False)
	par.add_argument("--matrix_nlayer", type = int , default = 4)

	#model structure 

	par.add_argument("--dropout"      , type = float , default = 0.0)
	par.add_argument("--loss" 		  , type = str , default = "loss_1")

	#training arguments
	par.add_argument("--t2g_batch_size" , type = int , default = 8)
	par.add_argument("--epoch_numb" 	, type = int , default = 50)
	par.add_argument("--t2g_lr" 		, type = float , default = 1e-4)
	par.add_argument("--warmup_prop"	, type = float , default = 0.1)
	par.add_argument("--scheduler" 		, type = str , default = "cosine") #linear or cosine

	par.add_argument("--ensemble_size" 	, type = int , default = 5)

	par.add_argument("--no_rel_weight" 		, type = float , default = 0.05)
	par.add_argument("--rel_weight_smooth" 	, type = float , default = 0)
	par.add_argument("--rel_weight_norm" 	, action = 'store_true')
	par.add_argument("--no_rel_name" 		, type = str , default = "NONE")

	#two-phase training
	par.add_argument("--binary" 		, action = "store_true" , default = False) # 只训练其判断正负例
	par.add_argument("--pos_only" 		, action = "store_true" , default = False) # 只训练其给正例分类

	#validattion and generation settings
	par.add_argument("--gene_no_rel" 	, action = "store_true" , default = False) # 评测时输出no_rel
	par.add_argument("--gene_in_data" 	, action = "store_true" , default = False) # 评测时只对测试集中出现的实体对生成
	par.add_argument("--valid_metric" 	, type = str   , default = "micro*macro")  # 用来选择最优模型的指标。因为 macro f1 常常不是很靠谱...
	par.add_argument("--pos_thresh" 	, type = float , default = 0.3) # 在两阶段生成中，正负例的阈值（以大于这个值的信心判断为正例，则认为是正例）

	#others 
	par.add_argument("--t2g_seed" 		, type = int , default = 2333)
	par.add_argument("--log_file" 		, type = str , default = "log.txt")
	par.add_argument("--no_log" 		, action = "store_true" , default = False)
	par.add_argument("--info" 			, type = str , default = "") # just to let fitlog record sth
	par.add_argument("--no_fitlog" 		, action = "store_true" , default = False) # 不使用fitlog
	par.add_argument("--no_valid" 		, action = "store_true" , default = False) # 是否使用验证集选择最好的参数来ensemble  

	#for analyze
	par.add_argument("--model_save" 	, type = str , default = "") # 保存最终的（ensemble的）模型的文件名
	par.add_argument("--gene_file" 		, type = str , default = "watch/gene") # 保存生成结果的文件
	par.add_argument("--watch_type" 	, type = str , default = "test") # 保存生成结果的文件
	par.add_argument("--model_save_2" 	, type = str , default = "") # 在两阶段生成中保存pos_only模型的文件

	#overall
	par.add_argument("--auto_hyperparam", action = "store_true" , default = False) # 自动确定合适的超参数

	#---------------------------------------------------------------------------------------------------


	return par

def auto_hyperparam(C):
	if C.dataset == "ace_2005":
		C.ensemble 		= 1 
		C.no_rel_name	= "NO_RELATION" 
		C.gnn  			= True
		C.matrix_trans  = True
		C.train_text_1	= "./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/bn+nw.json"
		C.valid_text	= "./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/bc_dev.json"
		C.test_text		= "./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/bc_test.json"
		C.dataset		= "ace_2005"
		C.gene_in_data 	= True
		C.valid_metric	= "macro"
		C.scheduler		= "cosine"
		C.no_valid 		= True
		C.loss 			= "loss_1"
		C.t2g_batch_size= 8 
		C.t2g_lr 		= 5e-5 
		C.no_rel_weight = 0.25 
		C.epoch_numb 	= 30  
		C.warmup_prop 	= 0.02
		C.model_save 	= "model_ace.pkl"
	elif C.dataset == "semeval_2018_task7":
		C.ensemble 		= 5 
		C.epoch_numb	= 30 
		C.no_rel_name 	= "NONE" 
		C.matrix_trans  = True
		C.gnn 			= True
		C.valid_text 	= "./data/semeval_2018_task7/2.test.text.xml"
		C.valid_rels 	= "./data/semeval_2018_task7/keys.test.2.txt"
		C.loss 			= "loss_2"
		C.no_valid 		= True
		C.warmup_prop 	= 0.1 
		C.scheduler 	= "cosine"
		C.t2g_batch_size= 8 
		C.t2g_lr 		= 1e-4
		C.model_save 	= "model_semeval.pkl"
	C.no_fitlog = True


def after_parse_t2g(C , need_logger = False):

	#----- make logger -----

	logger = Logger(C.log_file)
	logger.log = logger.log_print_w_time
	if C.no_log:
		logger.log = logger.nolog

	C.tmp_file_name = random_tmp_name()

	#----- other stuff -----

	if C.auto_hyperparam:
		auto_hyperparam(C)
		logger.log("Hyper parameters autoset.")

	if C.no_fitlog:
		fitlog.debug()

	fitlog.set_log_dir("logs")
	fitlog.add_hyper(C)

	logger.log ("------------------------------------------------------")
	logger.log (pformat(C.__dict__))
	logger.log ("------------------------------------------------------")

	C.gpus = list(range(tc.cuda.device_count()))


	#----- initialize -----

	if C.t2g_seed > 0:
		random.seed(C.t2g_seed)
		tc.manual_seed(C.t2g_seed)
		np.random.seed(C.t2g_seed)
		tc.cuda.manual_seed_all(C.t2g_seed)
		tc.backends.cudnn.deterministic = True
		tc.backends.cudnn.benchmark = False

		logger.log ("Seed set. %d" % (C.t2g_seed))

	tc.cuda.set_device(C.gpus[0])
	C.device = C.gpus[0]

	if need_logger:
		return C , logger

	return C

def get_config(): #not for t2g
	C , logger = after_parse_t2g(before_parse_t2g(argparse.ArgumentParser()).parse_args() , need_logger = True)


	C.batch_size = C.t2g_batch_size
	C.lr = C.t2g_lr

	return C , logger