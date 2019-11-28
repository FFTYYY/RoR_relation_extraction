import argparse
import os
from utils.logger import Logger
from pprint import pformat
from fastNLP.core._logger import logger as inner_logger
import random
import torch as tc
import numpy as np

_par = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#dataloader
_par.add_argument("--data_path" 	, type = str , default = "./semeval_2018_task7/")
_par.add_argument("--force_reprocess" 	, action = "store_true", default = False)

_par.add_argument("--test_mode" 	, action = "store_true" , default = False)

#others
_par.add_argument("--gpus" 			, type = str , default = "0")
_par.add_argument("--seed" 			, type = int , default = 2333)
_par.add_argument("--log_file" 		, type = str , default = "log.txt")
_par.add_argument("--no_log" 		, action = "store_true" , default = False)

#---------------------------------------------------------------------------------------------------

C = _par.parse_args()

def listize(name):
	C.__dict__[name] = [int(x) for x in filter(lambda x:x , C.__dict__[name].strip().split(","))]

listize("gpus")

if C.test_mode:
	C.log_file += ".test"

logger = Logger(inner_logger , C.log_file)
logger.log = logger.log_print_w_time
if C.no_log:
	logger.log = logger.nolog

logger.log ("------------------------------------------------------")
logger.log (pformat(C.__dict__))
logger.log ("------------------------------------------------------")


#Initialize

if C.seed > 0:
	random.seed(C.seed)
	tc.manual_seed(C.seed)
	np.random.seed(C.seed)
	tc.cuda.manual_seed_all(C.seed)
	tc.backends.cudnn.deterministic = True
	tc.backends.cudnn.benchmark = False

	logger.log ("Seed set. %d" % (C.seed))

tc.cuda.set_device(C.gpus[0])