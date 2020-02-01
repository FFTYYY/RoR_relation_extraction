import argparse
import os
from utils.logger import Logger
from pprint import pformat
import random
import torch as tc
import numpy as np
from utils.tmp_file import random_tmp_name
import fitlog

#fitlog.commit(__file__)

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
	par.add_argument("--model" 		, type = str , default = "naive_bert" , 
		choices = [
			"naive_bert" , 
			"graph_trans" , 
	])

	#model structure 

	par.add_argument("--dropout"      , type = float , default = 0.0)
	par.add_argument("--loss" 		, type = str , default = "loss_3")

	#training arguments
	par.add_argument("--t2g_batch_size" , type = int , default = 8)
	par.add_argument("--epoch_numb" 	, type = int , default = 50)
	par.add_argument("--t2g_lr" 		, type = float , default = 1e-4)
	par.add_argument("--fine_tune" 		, action = "store_true" , default = False)
	par.add_argument("--warmup_prop"	, type = float , default = 0.1)
	par.add_argument("--scheduler" 		, type = str , default = "cosine") #linear or cosine

	par.add_argument("--ensemble_size" 	, type = int , default = 5)

	par.add_argument("--no_rel_weight" 		, type = float , default = 0.05)
	par.add_argument("--rel_weight_smooth" 	, type = float , default = 0)
	par.add_argument("--rel_weight_norm" 	, action = 'store_true')
	par.add_argument("--no_rel_name" 		, type = str , default = "NONE") # 只训练其判断正负例

	#two-phase training
	par.add_argument("--binary" 		, action = "store_true" , default = False) # 只训练其判断正负例
	par.add_argument("--pos_only" 		, action = "store_true" , default = False) # 只训练其给正例分类

	#validattion and generation settings
	par.add_argument("--gene_no_rel" 	, action = "store_true" , default = False) # 评测时输出no_rel
	par.add_argument("--gene_in_data" 	, action = "store_true" , default = False) # 评测时只对测试集中出现的实体对生成
	par.add_argument("--valid_metric" 	, type = str   , default = "micro*macro")  # 用来选择最优模型的指标。因为 macro f1 常常不是很靠谱...
	par.add_argument("--pos_thresh" 	, type = float , default = 0.3) # 在两阶段生成中，正负例的阈值（以大于这个值的信心判断为正例，则认为是正例）

	#others 
	par.add_argument("--gpus" 			, type = str , default = "0")
	par.add_argument("--t2g_seed" 		, type = int , default = 2333)
	par.add_argument("--log_file" 		, type = str , default = "log.txt")
	par.add_argument("--no_log" 		, action = "store_true" , default = False)
	par.add_argument("--info" 			, type = str , default = "") # just to let fitlog record sth
	par.add_argument("--no_fitlog" 		, action = "store_true" , default = False) # 不使用fitlog
	par.add_argument("--no_valid" 		, action = "store_true" , default = False) # 是否使用验证集选择最好的参数来ensemble  

	# for watch
	par.add_argument("--model_save" 	, type = str , default = "") # 保存最终的（ensemble的）模型的文件名
	par.add_argument("--gene_file" 		, type = str , default = "watch/gene") # 保存生成结果的文件
	par.add_argument("--watch_type" 	, type = str , default = "test") # 保存生成结果的文件
	par.add_argument("--model_save_2" 	, type = str , default = "") # 在两阶段生成中保存pos_only模型的文件


	#---------------------------------------------------------------------------------------------------


	return par

def after_parse_t2g(C , need_logger = False):

	#----- do some check -----

	for file in [C.train_text_1 , C.train_rels_1 , C.train_text_2 , C.train_rels_2 , C.test_text , C.test_rels ]:
		if C.dataset not in file:
			print('[Warn] Dataset and training files do not match.')
			#import pdb;pdb.set_trace()

	#----- make logger -----

	logger = Logger(C.log_file)
	logger.log = logger.log_print_w_time
	if C.no_log:
		logger.log = logger.nolog

	C.tmp_file_name = random_tmp_name()

	#----- other stuff -----

	if C.no_fitlog:
		fitlog.debug()

	fitlog.set_log_dir("logs")
	fitlog.add_hyper(C)

	logger.log ("------------------------------------------------------")
	logger.log (pformat(C.__dict__))
	logger.log ("------------------------------------------------------")

	#fitlog不支持list，所以把字符串作为参数，之后再转成list
	C.gpus = [int(x) for x in C.gpus.strip().split(",")] 

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