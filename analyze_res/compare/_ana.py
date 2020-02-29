import json
import pdb

config = 0

#---------------------------------------------------------------------------------------------------
#configs

if config == 0: # semeval
	file_name = "semeval-test.json"
	oracle_deg = 1
	no_rel = "NONE"


if config == 1: # ace bctest
	file_name = "ace2005-bctest.json"
	oracle_deg = None
	no_rel = "NO_RELATION"


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

for model_id in [1,2]:
	avg_nr_diff 		= 0 # E |nr(golden) - nr(output)|
	avg_max_deg_diff 	= 0 # E| 1 - max(deg(output)) |
	avg_node_exceed_or 	= 0 # E| num of node that degree > 1|

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


		avg_nr_diff += abs(nr_m - nr_g)

		_odeg = oracle_deg or max_deg_g
		avg_max_deg_diff += max(max_deg_m - _odeg , 0 )

		avg_node_exceed_or += sum( [int(d > _odeg) for d in deg_m] )

	avg_node_exceed_or 	/= len(com_res)
	avg_nr_diff 		/= len(com_res)
	avg_max_deg_diff 	/= len(com_res)

	print ("model %d: avg_nr_diff = %.4f , avg_max_deg_diff = %.4f avg_node_exceed_or = %.4f" % (
		model_id , 
		avg_nr_diff , 
		avg_max_deg_diff , 
		avg_node_exceed_or , 
	))


