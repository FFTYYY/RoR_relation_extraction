import torch as tc
import fitlog

def pad_sents(sents , pad_idx = 0):
	max_len = max([len(x) for x in sents])
	for i in range(len(sents)):
		sents[i] += [pad_idx] * (max_len - len(sents[i]))
	return sents

def get_data_from_batch(data, device=tc.device(0)):
	sents = [x.abs  for x in data] 
	ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
	anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]
	data_ent = [x.ents for x in data] 
	sents = pad_sents(sents)
	sents = tc.LongTensor(sents).to(device)

	return sents , ents , anss , data_ent

