'''
	一些指标
'''

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

for model_id in [1,2]:
	
	avg_nr_diff 		= 0 # E |nr(golden) - nr(output)|
	avg_max_deg_diff 	= 0 # E| 1 - max(deg(output)) |
	avg_node_exceed_or 	= 0 # E| num of node that degree > 1|
	avg_deg_dist_diff 	= 0

	zero_rel = 0
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

		if model_id == 1:
			got_ans = got_ans_1
		elif model_id == 2:
			got_ans = got_ans_2
		else:
			assert False

		deg_m = get_degree(got_ans , ent_num)
		deg_g = get_degree(golden_ans , ent_num)

		nr_m , max_deg_m = get_index(deg_m)
		nr_g , max_deg_g = get_index(deg_g)

		if max_deg_g == 0 and max_deg_m == 0:
			zero_rel += 1
			if ignore_zero_rel:
				continue

		avg_nr_diff += abs(nr_m - nr_g)

		_odeg = oracle_deg or max_deg_g
		avg_max_deg_diff += max(max_deg_m - _odeg , 0 )

		avg_node_exceed_or += sum( [int(d > _odeg) for d in deg_m] )

		avg_deg_dist_diff += sum ( [ abs(deg_m[i] - deg_g[i]) for i in range(ent_num)] )

	tot_num = len(com_res)
	if ignore_zero_rel:
		tot_num -= zero_rel

	avg_node_exceed_or 	/= tot_num
	avg_nr_diff 		/= tot_num
	avg_max_deg_diff 	/= tot_num
	avg_deg_dist_diff 	/= tot_num

	print ("model %d: avg_nr_diff = %.4f , avg_max_deg_diff = %.4f avg_node_exceed_or = %.4f avg_deg_dist_diff = %.4f" % (
		model_id , 
		avg_nr_diff , 
		avg_max_deg_diff , 
		avg_node_exceed_or , 
		avg_deg_dist_diff , 
	))

print()

co_ocus = {}
for model_id in [0,1,2]:
	
	co_ocu 				= {}

	zero_rel = 0
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

		if model_id == 1:
			got_ans = got_ans_1
		elif model_id == 2:
			got_ans = got_ans_2
		elif model_id == 0:
			got_ans = golden_ans
		else:
			assert False

		if max_deg_g == 0 and max_deg_m == 0:
			zero_rel += 1
			if ignore_zero_rel:
				continue

		for u1,v1,t1 in got_ans:
			if t1 == no_rel:
				continue
			rel2id(t1)
			for u2,v2,t2 in got_ans:
				if t2 == no_rel:
					continue
				if (u1,v1) == (u2,v2):
					continue
				co_ocu[ (t1,t2) ] = co_ocu.get((t1,t2) , 0) + 1 

	tot_num = len(com_res)
	if ignore_zero_rel:
		tot_num -= zero_rel

	for x in co_ocu:
		co_ocu[x] /= tot_num

	co_ocus[model_id] = co_ocu

	co_ocu = [ [i] + [ "%.2f" % (co_ocu.get((i,j),0)) for j in rels ] for i in rels]
	co_ocu_s = beautiful_str([" "] + rels , co_ocu)


	print ("co ocu matrix of model %d:" % (model_id))
	print (co_ocu_s)

