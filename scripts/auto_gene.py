import os , sys
import pdb

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))

ace_script = """python -m analyze.gene 
--dataset=ace_2005 --gene_no_rel --gene_in_data --no_rel_name=NO_RELATION 
--train_text_1=./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/bn+nw.json 
--valid_text=./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/bc_dev.json 
--test_text=./data/ace_2005/ace_05_processed/ace-05-splits/json-pm13/{split_file}.json 
--watch_type=test --model_save={setting_file} 
--gene_file=analyze_res/compare-ace05/ace2005-{split_name}-{setting_name}"""

# [split_name , split_file]
ace_splits = [
	["bcdev" 	, "bc_dev"	] ,
	["bctest" 	, "bc_test"	] ,
	["wl" 		, "wl"		] ,
	["cts" 		, "cts"		] , 
	["train" 	, "bn+nw"	] ,
]


# [setting_name , setting_file]
ace_models = [
	["4"		, "model_r4.pkl"] , # baseline
	["5"		, "model_r2.pkl"] , # gnn
	["6"		, "model_r3.pkl"] , # matrix
	["7"		, "model_r1.pkl"] , # gnn+matrix
	["golden"	, "golden" 		] , # golden
]


semeval_script = """python -m analyze.gene 
--dataset=semeval_2018_task7 --gene_no_rel --no_rel_name=NONE 
--model_save={setting_file} 
--watch_type={split_file} --gene_file=analyze_res/compare-semeval18/semeval-{split_name}-{setting_name}"""

semeval_splits = [
	["train" , "test"] , 
	["train" , "train"] ,
]

semeval_models = [
	["4"		, "sem_model_r6.pkl"] , # baseline
	["5"		, "sem_model_r3.pkl"] , # gnn
	["6"		, "sem_model_r2.pkl"] , # matrix
	["7"		, "sem_model_re.pkl"] , # gnn+matrix
	["golden"	, "golden" 		] , # golden
]


def run(script , splits , settings):
	for split in splits:
		for setting in settings:
			now_script = script.format(
				split_name = split[0] , split_file = split[1] , 
				setting_name = setting[0] , setting_file = setting[1]
			)

			now_script = now_script.replace("\n" , " ")
			now_script = now_script + " --no_log"

			print ("Split %s and Setting %s" % (split[0] , setting[0]))
			os.system(now_script)
			print ()


if __name__ == "__main__":
	#run(ace_script , ace_splits , ace_models)	
	run(semeval_script , semeval_splits , semeval_models)