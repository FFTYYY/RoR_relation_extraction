

def write_keyfile(data , generator):
	'''
	将data写成 semeval 2018 task7 key file的格式
	entity名：doc_id.entity_number
	'''

	relations = generator.relations
	#gene_no_rel = generator.gene_no_rel
	#no_rel = generator.no_rel

	cont = ""
	for x in data:

		got_relations = [] # format: (u,v) for u < v

		for r in x.ans:
			u,v = r.u,r.v
			
			reverse = False
			if u > v:
				u,v = v,u
				reverse = True

			got_relations.append((u,v))

			u,v = x.id2ent_name(u) , x.id2ent_name(v)
			cont += "%s(%s,%s%s)\n" % (relations[r.type] , u , v , ",REVERSE" if reverse else "")

		#if gene_no_rel:
		#	for i in range(len(x.ents)):
		#		for j in range(i):
		#			if (j,i) not in got_relations:
		#				cont += "%s(%s,%s)\n" % (relations[no_rel] , x.ents[j].name , x.ents[i].name)


	return cont