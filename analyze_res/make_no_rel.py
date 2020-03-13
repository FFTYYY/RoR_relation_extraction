import os
import json
import pdb

old_dirname = "compare-semeval18/"
new_dirname = "compare-semeval18-norel-unified/"

os.makedirs(new_dirname , exist_ok = True)

for filename in list(os.walk(old_dirname))[0][2]:
	if not filename.endswith(".json"):
		continue
	old_filename = os.path.join(old_dirname , filename)
	new_filename = os.path.join(new_dirname , filename)

	with open(old_filename , "r") as fil:
		c = json.load(fil)

	for x in c:
		for y in x["relations"]:
			if y[2] == "NONE":
				y[2] = "NO_RELATION"

	with open(new_filename , "w") as fil:
		json.dump(x , fil)

