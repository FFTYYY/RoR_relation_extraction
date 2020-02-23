def intize(lis , ks):
	'''
		lis的每个元素也是list
		将lis的每个元素的ks位置的元素转为int
	'''

	ret = []
	for x in lis:
		nx = []
		for i in range(len(x)):
			if i in ks:
				nx.append(int(x[i]))
			else:
				nx.append(x[i])
		ret.append(nx)
	return ret