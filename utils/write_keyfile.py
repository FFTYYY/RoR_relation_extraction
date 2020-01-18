

def write_keyfile(data , generator):
	'''
	将data写成 semeval 2018 task7 key file的格式
	entity名：doc_id.entity_number
	'''

	relations = generator.relations

	cont = ""
	for x in data:
		for r in x.ans:
			u,v = r.u,r.v
			
			reverse = False
			if u > v:
				u,v = v,u
				reverse = True

			u,v = x.id2ent_name(u) , x.id2ent_name(v)
			cont += "%s(%s,%s%s)\n" % (relations[r.type] , u,v,",REVERSE" if reverse else "")
	return cont