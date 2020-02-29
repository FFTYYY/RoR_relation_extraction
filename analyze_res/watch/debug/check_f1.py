from sklearn.metrics import precision_score , recall_score , f1_score , confusion_matrix , classification_report

relations = []

def ask_index(rel_name):
	if not rel_name in relations:
		relations.append(rel_name)
	return relations.index(rel_name)

gene = []
gold = []

with open("gene.txt" , "r") as fil:
	for line in fil:
		line = line.strip()
		if line == "":
			continue
		rel_name = line.split("(")[0]
		gene.append(ask_index(rel_name))

with open("golden.txt" , "r") as fil:
	for line in fil:
		line = line.strip()
		if line == "":
			continue
		rel_name = line.split("(")[0]
		gold.append(ask_index(rel_name))


f1 = f1_score(gold , gene , average = "macro" , labels = range(1,7))
p  = precision_score(gold , gene , average = "macro" , labels = range(1,7))
r  = recall_score(gold , gene , average = "macro" , labels = range(1,7))

print (f1)
print ((2*p*r) / (p + r))


p  = classification_report(gold , gene , digits = 4 , labels = range(1,7))
print (p)
