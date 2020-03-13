'''
得出heatmap，并绘图
'''

import json
import pdb
from YTools.universe.beautiful_str import beautiful_str
import torch as tc
import visdom
from copy import deepcopy

viz = visdom.Visdom(env = 'main')

config = 4

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

if config == 2: # ace wl
	file_name = "ace2005-wl.json"
	oracle_deg = None
	no_rel = "NO_RELATION"

if config == 3: # ace cts
	file_name = "ace2005-cts.json"
	oracle_deg = None
	no_rel = "NO_RELATION"

if config == 4: # ace dev
	file_name = "ace2005-bcdev.json"
	oracle_deg = None
	no_rel = "NO_RELATION"

if config == 5: # ace all
	file_name = ["ace2005-bctest.json" , "ace2005-wl.json" , "ace2005-cts.json"]
	oracle_deg = None
	no_rel = "NO_RELATION"

#---------------------------------------------------------------------------------------------------

if isinstance(file_name , list):
	com_res = []
	for x in file_name:
		with open(x , "r") as fil:
			com_res = com_res + json.load(fil)
else: 
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

to_draw = []
model_names = ["golden" , "baseline" , "sota"]
#只要出现就算，无视方向
nd_co_ocus = {}
for model_id in [0,1,2]:
	
	nd_co_ocu_ii = {}
	nd_co_ocu_oo = {}
	nd_co_ocu_io = {}

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

		nodes = set()
		nd_rels_o = {}
		nd_rels_i = {}
		for u,v,t in got_ans:
			if t == no_rel:
				continue

			if not nd_rels_o.get(u):
				nd_rels_o[u] = []
			if not nd_rels_i.get(v):
				nd_rels_i[v] = []
			nd_rels_o[u].append(t)
			nd_rels_i[v].append(t)

			nodes.add(u)
			nodes.add(v)

			rel2id(t)

		for nd in nodes:
			for i , r1 in enumerate(nd_rels_i.get(nd , [])):
				for j , r2 in enumerate(nd_rels_i.get(nd , [])):
					if i == j:
						continue
					nd_co_ocu_ii[(r1,r2)] = nd_co_ocu_ii.get((r1,r2),0) + 1
			for i , r1 in enumerate(nd_rels_o.get(nd , [])):
				for j , r2 in enumerate(nd_rels_o.get(nd , [])):
					if i == j:
						continue
					nd_co_ocu_oo[(r1,r2)] = nd_co_ocu_oo.get((r1,r2),0) + 1
			for i , r1 in enumerate(nd_rels_i.get(nd , [])):
				for j , r2 in enumerate(nd_rels_o.get(nd , [])):
					nd_co_ocu_io[(r1,r2)] = nd_co_ocu_io.get((r1,r2),0) + 1
			for i , r1 in enumerate(nd_rels_o.get(nd , [])):
				for j , r2 in enumerate(nd_rels_i.get(nd , [])):
					nd_co_ocu_io[(r1,r2)] = nd_co_ocu_io.get((r1,r2),0) + 1

	tot_num = len(com_res)

	#for x in nd_co_ocu:
	#	nd_co_ocu[x] = 100 * nd_co_ocu[x] / len(com_res)

	#nd_co_ocus[model_id] = nd_co_ocu

	rr_names = ["I-I" , "O-O" , "I-O"]
	for rr_id , nd_co_ocu in enumerate([nd_co_ocu_ii , nd_co_ocu_oo , nd_co_ocu_io]):

		nd_co_ocu = tc.Tensor([ [ nd_co_ocu.get((i,j) , 0) for j in rels] for i in rels])
		nd_co_ocu = nd_co_ocu / nd_co_ocu.sum()

		nd_co_ocus[model_names[model_id] + " " + rr_names[rr_id]] = nd_co_ocu
		#nd_co_ocu = tc.Tensor(nd_co_ocu)

		#nd_co_ocu = [ [i] + [ "%d" % (nd_co_ocu.get((i,j),0)) for j in rels ] for i in rels]
		#nd_co_ocu_s = beautiful_str([" "] + rels , nd_co_ocu)

		def draw_func(x , rels , name_1 , name_2):
			return lambda : viz.heatmap(
				X = x,
				opts = dict(
					colormap = "Electric",
					rownames = rels , 
					columnnames = rels, 
					title = "%s rels model %s" % (name_1 , name_2) , 
				)
			)
		to_draw.append(draw_func(nd_co_ocu , rels , rr_names[rr_id] , model_names[model_id]))

	#print ("co ocu matrix of model %d:" % (model_id))
	#print (nd_co_ocu_s)

print ("I-I baseline %.2f" % ((nd_co_ocus["baseline I-I"] - nd_co_ocus["golden I-I"]) ** 2).sum() ** 0.5)
print ("I-I sota     %.2f" % ((nd_co_ocus["sota I-I"]     - nd_co_ocus["golden I-I"]) ** 2).sum() ** 0.5)
print ("O-O baseline %.2f" % ((nd_co_ocus["baseline O-O"] - nd_co_ocus["golden O-O"]) ** 2).sum() ** 0.5)
print ("O-O sota     %.2f" % ((nd_co_ocus["sota O-O"]     - nd_co_ocus["golden O-O"]) ** 2).sum() ** 0.5)
print ("I-O baseline %.2f" % ((nd_co_ocus["baseline I-O"] - nd_co_ocus["golden I-O"]) ** 2).sum() ** 0.5)
print ("I-O sota     %.2f" % ((nd_co_ocus["sota I-O"]     - nd_co_ocus["golden I-O"]) ** 2).sum() ** 0.5)

#to_draw[0]()
#to_draw[3]()
#to_draw[6]()
#
#to_draw[1]()
#to_draw[4]()
#to_draw[7]()
#
#to_draw[2]()
#to_draw[5]()
#to_draw[8]()