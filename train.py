from config import C , logger
from dataloader_subt2 import run as read_data , relations , id2rel
from tqdm import tqdm
from utils.train_util import pad_sents , pad_ents , pad_anss
import torch as tc
from models import models
import pdb
import os , sys
import math


def load_data():
	data_train , data_test = read_data(C.train_text , C.train_rels , C.test_text , C.test_rels)

	return data_train , data_test

def valid(dataset , model , epoch_id = 0):

	model = model.eval()

	batch_numb = (len(dataset) // C.batch_size) + int((len(dataset) % C.batch_size) != 0)
	pbar = tqdm(range(batch_numb) , ncols = 70)
	avg_loss = 0
	res_file = open("./tmp.txt" , "w" , encoding = "utf-8")
	for batch_id in pbar:
		data = dataset[batch_id * C.batch_size : (batch_id+1) * C.batch_size]

		sents = [x.abs  for x in data] 
		ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
		anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]
		data_ent = [x.ents for x in data] 

		sents 	= pad_sents(sents)
		sents = tc.LongTensor(sents).cuda()

		with tc.no_grad():
			pred = model(sents , ents)
			loss = model.loss(pred , anss , ents)
			#loss = 0.
			model.generate(pred , data_ent , id2rel , res_file)

		try:
			assert not math.isnan(float(loss))
		except AssertionError:
			pdb.set_trace()

		avg_loss += float(loss)

		pbar.set_description_str("(Test )Epoch %d" % (epoch_id + 1))
		pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))

	res_file.close()
	os.system("perl {script} {result_file} {key_file}".format(
		script 		= C.test_script ,
		result_file = "./tmp.txt" ,
		key_file 	= C.test_rels ,
	))	
	print ("Epoch %d tested. avg_loss = %.4f" % (epoch_id + 1 , avg_loss / batch_numb))


	model = model.train()
	#pdb.set_trace()

def train(train_data , test_data):

	model = models[C.model](relation_typs = len(relations) + 1 , dropout = C.dropout).cuda()
	optimizer = tc.optim.Adam(params = model.parameters() , lr = C.lr)

	batch_numb = (len(train_data) // C.batch_size) + int((len(train_data) % C.batch_size) != 0)
	for epoch_id in range(C.epoch_numb):

		pbar = tqdm(range(batch_numb) , ncols = 70)
		avg_loss = 0
		for batch_id in pbar:
			data = train_data[batch_id * C.batch_size : (batch_id+1) * C.batch_size]

			sents = [x.abs  for x in data] 
			ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
			anss  = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]

			sents = pad_sents(sents)
			sents = tc.LongTensor(sents).cuda()

			pred = model(sents , ents)
			loss = model.loss(pred , anss , ents)

			try:
				assert loss.item() == loss.item()
			except AssertionError:
				pdb.set_trace()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			avg_loss += float(loss)

			pbar.set_description_str("(Train)Epoch %d" % (epoch_id + 1))
			pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))
		print ("Epoch %d ended. avg_loss = %.4f" % (epoch_id + 1 , avg_loss / batch_numb))
		valid(test_data , model , epoch_id)

	return model

if __name__ == "__main__":

	data_train , data_test = load_data()

	model = train(data_train , data_test)