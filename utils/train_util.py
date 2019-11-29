
def pad_sents(sents , pad_idx = 0):
	max_len = max([len(x) for x in sents])
	for i in range(len(sents)):
		sents[i] += [pad_idx] * (max_len - len(sents[i]))
	return sents

def pad_ents(ents , pad_idx = -1):
	return ents #no padding

def pad_anss(anss , pod_idx = -1):
	return anss #no padding
