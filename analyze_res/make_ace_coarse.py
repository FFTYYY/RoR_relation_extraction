import json
import os
import pdb

ace_filename = "ace2005-{split_name}-{setting_name}.json"
old_path = "compare-ace05-fine/"
new_path = "compare-ace05-coarse/"

os.makedirs(new_path , exist_ok = True)

splits = [
	["train" 	, ["train"]] , 
	["dev" 		, ["bcdev"]] , 
	["test" 	, ["bctest" , "wl" , "cts"]] , 
]

settings = ["4" , "5" , "6" , "7" , "golden"]

for setting in settings:
	for name , filenames in splits:
		new_file_name = os.path.join(
			new_path , 
			ace_filename.format(split_name = name , setting_name = setting)
		)

		new_file_content = []
		for f in filenames:
			old_file_name = os.path.join(
				old_path , 
				ace_filename.format(split_name = f , setting_name = setting)
			)
			with open(old_file_name , "r") as fil:
				cont = json.load(fil)
				new_file_content = new_file_content + cont


		with open(new_file_name , "w") as fil:
			json.dump(new_file_content , fil)