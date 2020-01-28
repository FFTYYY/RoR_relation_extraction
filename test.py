from tqdm import tqdm
import torch as tc
import pdb
import os , sys
import math
import fitlog
import re
from utils.scorer import get_f1
from utils.train_util import pad_sents , get_data_from_batch
from utils.write_keyfile import write_keyfile

def before_test(C , logger , dataset , models):
	
	if isinstance(models , tc.nn.Module):
		models = [models]
	for i in range(len(models)):
		models[i] = models[i].eval()

	device = tc.device(C.device)
	batch_size = 8
	batch_numb = (len(dataset) // batch_size) + int((len(dataset) % batch_size) != 0)

	return device , batch_size , batch_numb , models

def get_output(C , logger , 
		models , device , loss_func , generator , 
		sents , ents , anss , data_ent , 
	):


	preds = [0 for _ in range(len(models))]
	for i , model in enumerate(models):

		old_device = next(model.parameters()).device
		model = model.to(device)
		preds[i] = model(sents , ents)
		model = model.to(old_device) #如果他本来在cpu上，生成完之后还是把他放回cpu

		loss = loss_func(preds[i] , anss , ents)

	ans_rels = [ [(u,v) for u,v,t in bat] for bat in anss] if C.rel_only else None
	generated = generator(preds , data_ent , ans_rels = ans_rels)

	#pred_map = pred.max(-1)[1] #(ne , ne)

	return model , preds , loss , generated

def get_evaluate(C , logger , mode , generated , generator , test_data = None):

	golden = write_keyfile(test_data , generator)

	#pdb.set_trace()

	micro_f1 , macro_f1 = get_f1(golden , generated , is_file_content = True , no_rel = generator.get_no_rel_name())
	micro_f1 , macro_f1 = micro_f1 * 100 , macro_f1 * 100

	os.makedirs("watch/debug" , exist_ok = True)
	with open("watch/debug/golden.txt" , "w") as fil:
		fil.write(golden)
	with open("watch/debug/gene.txt" , "w") as fil:
		fil.write(generated)
	#pdb.set_trace()


	return micro_f1 , macro_f1


def test(C , logger , 
		dataset , models , 
		loss_func , generator , 
		mode = "valid" , epoch_id = 0 , run_name = "0" , need_generated = False , 
	):
		

	_step = 0
	if epoch_id == 1 or epoch_id == 16:
		debug_file = open("watch/debug/epoch_{0}.txt".format(epoch_id) , "w")
	else:
		debug_file = None


	device , batch_size , batch_numb , models = before_test(C , logger , dataset , models)

	pbar = tqdm(range(batch_numb) , ncols = 70)
	avg_loss = 0
	generated = ""
	for batch_id in pbar:


		data = dataset[batch_id * batch_size : (batch_id+1) * batch_size]
		sents , ents , anss , data_ent = get_data_from_batch(data, device=tc.device(C.device))

		with tc.no_grad():
			model , preds , loss , partial_generated = get_output(
				C,logger,models,device,loss_func,generator,sents,ents,anss,data_ent
			)
		generated += partial_generated
		avg_loss += float(loss) / len(models)

		# debug
		if (not debug_file is None) and (batch_id <= 5):
			prd = preds[0]
			for _b in range(len(prd)):
				chooses = []
				goldens = []

				pred_vectors = []
				the_loss = 0.
				good = 0
				for _i,_j,golden in anss[_b]:
					choose = prd[_b,_i,_j].max(-1)[1]

					chooses.append(int(choose))
					goldens.append(int(golden))
					good += int(int(golden) == int(choose)) 
					pred_vectors.append([
						_i , _j , int(golden) , int(choose) , 
						"%.4f" % (float(prd[_b,_i,_j,golden])) , "%.4f" % (float(prd[_b,_i,_j,choose])) , 
						"%.4f" % (-math.log(float(prd[_b,_i,_j,golden]))) , int(int(golden) == int(choose)) 
					])
					the_loss -= math.log(float(prd[_b,_i,_j,golden]))
				from YTools.universe.beautiful_str import beautiful_str
				from sklearn.metrics import f1_score , precision_score , recall_score

				#----- text id -----

				debug_file.write("text_%d:\n" % _step)

				#----- sample f1 -----
				f1_micro = f1_score(goldens, chooses, average='micro', labels = [0,1,2,3,4,5] , zero_division=0)

				preci = precision_score(goldens, chooses, average='macro', labels = [0,1,2,3,4,5] , zero_division=0)
				recal =    recall_score(goldens, chooses, average='macro', labels = [0,1,2,3,4,5] , zero_division=0)
				debug_file.write ("precision / recall = %.4f , %.4f\n" % (float(preci) , float(recal)))
				if (preci + recal) == 0:
					f1_macro = 0
				else:
					f1_macro = (2 * preci * recal) / (preci + recal)
				debug_file.write("f1_micro / f1_macro = %.4f , %.4f\n" % (float(f1_micro) , float(f1_macro)))
				#----- loss -----

				debug_file.write("loss: %.4f / %d = %.4f \n" % (
					the_loss , len(anss[_b]) , the_loss / len(anss[_b])
				))
				debug_file.write("acc: %d/%d = %.4f \n\n" % (
					good , len(anss[_b]) , good / len(anss[_b])
				))

				#----- details -----

				debug_file.write(beautiful_str(
					["u" , "v" , "golden" , "model choice" , "golden prob" , "model choice prob" , "loss contribution" , "good"] , 
					pred_vectors , 
				))

				debug_file.write("\n\n")
				_step += 1
			debug_file.write ("\n\n now batch loss = %.4f , avg loss = %.4f , for validation.\n" % 
				(float(loss) , avg_loss / (batch_id+1)))

		pbar.set_description_str("(Test )Epoch {0}".format(epoch_id))
		pbar.set_postfix_str("loss = %.4f (avg = %.4f)" % ( float(loss) , avg_loss / (batch_id+1)))

	micro_f1 , macro_f1 = get_evaluate(C , logger , mode , generated , generator , dataset)

	#print (result)
	logger.log ("-----Epoch {} tested. Micro F1 = {:.2f}% , Macro F1 = {:.2f}% , loss = {:.4f}".
			format(epoch_id , micro_f1, macro_f1, avg_loss / batch_numb))
	logger.log("\n")

	fitlog.add_metric(micro_f1 , step = epoch_id , name = "({0})micro f1".format(run_name)) 
	fitlog.add_metric(macro_f1 , step = epoch_id , name = "({0})macro f1".format(run_name)) 

	if need_generated:
		return micro_f1 , macro_f1 , avg_loss , generated

	return micro_f1 , macro_f1 , avg_loss