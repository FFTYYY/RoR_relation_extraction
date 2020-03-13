import json
import pdb
from YTools.universe.beautiful_str import beautiful_str
config = 2

#---------------------------------------------------------------------------------------------------
#configs

if config == 0: # semeval
	file_name = "semeval-test.json"
	oracle_deg = 1
	no_rel = "NONE"
	ignore_zero_rel = False


if config == 1: # ace bctest
	file_name = "ace2005-bctest.json"
	oracle_deg = None
	no_rel = "NO_RELATION"
	ignore_zero_rel = False

if config == 2: # ace wl
	file_name = "ace2005-wl.json"
	oracle_deg = None
	no_rel = "NO_RELATION"
	ignore_zero_rel = False

#---------------------------------------------------------------------------------------------------

with open(file_name , "r") as fil:
	com_res = json.load(fil)

def get_degree(rs , ent_num):
	deg = [0 for _ in range(ent_num)]

	for x,y,t in rs:
		if t == no_rel:
			continue
		deg[x] += 1
		deg[y] += 1
	return deg

def get_index(deg):
	nr = sum(deg)
	max_deg = max(deg)
	return nr , max_deg

rels = []
def rel2id(rel):
	if not rel in rels:
		rels.append(rel)
	return rels.index(rel)
def id2rel(id):
	return rels[id]


for x in com_res:
	ent_num = len(x["entitys"])

	if "relations" in x:
		golden_ans 	= [[y[0],y[1],y[2]] for y in x["relations"]]
		got_ans_1 	= [[y[0],y[1],y[3]] for y in x["relations"]]
		got_ans_2 	= [[y[0],y[1],y[4]] for y in x["relations"]]
	else:
		golden_ans 	= x["golden_ans"]
		got_ans_1 	= x["got_ans_1"]
		got_ans_2 	= x["got_ans_2"]

	deg_m = get_degree(got_ans , ent_num)



