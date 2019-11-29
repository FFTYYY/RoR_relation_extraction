from config import C , logger
from dataloader_subt2 import run as read_data , relations
from tqdm import tqdm
from utils.train_util import pad_sents , pad_ents , pad_anss
import torch as tc
from models import models
import pdb


def load_data():
	data_train , data_test = read_data(C.data_path)

	return data_train , data_test

def train(dataset):

	model = models[C.model](relation_typs = len(relations) + 1).cuda()
	optimizer = tc.optim.Adam(params = model.parameters() , lr = C.lr)

	batch_numb = (len(dataset) // C.batch_size) + int((len(dataset) % C.batch_size) != 0)
	for epoch_id in range(C.epoch_numb):

		pbar = tqdm(range(batch_numb) , ncols = 70)
		avg_loss = 0
		for batch_id in pbar:
			data = dataset[batch_id * C.batch_size : (batch_id+1) * C.batch_size]

			sents = [x.abs  for x in data] 
			ents  = [[ [e.s , e.e] for e in x.ents ] for x in data] 
			anss   = [[ [a.u , a.v , a.type] for a in x.ans ] for x in data]

			sents 	= pad_sents(sents)
			ents 	= pad_ents(ents)
			anss 	= pad_anss(anss)

			sents = tc.LongTensor(sents).cuda()

			pred = model(sents , ents)

			loss = model.loss(pred , anss , ents)

			try:
				assert loss.item() == loss.item()
			except Exception:
				pdb.set_trace()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			avg_loss += float(loss)

			pbar.set_description_str("Epoch %d" % (epoch_id + 1))
			pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))
		print ("Epoch %d ended. avg_loss = %.4f" % (epoch_id + 1 , avg_loss / batch_numb))

	return model

if __name__ == "__main__":

	data_train , data_test = load_data()

	model = train(data_train)