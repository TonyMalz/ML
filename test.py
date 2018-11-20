
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log

data = pd.read_csv("test.csv")

target = 'worthy'
if not target in data.columns:
	print('Target "%s" not found in dataset' % target)
	exit(1)

def entropy(counts):
	sum = counts.sum()
	H = 0
	for i in range(0,len(counts)):
		label = counts.index[i]
		count = counts[i]
		#print(count,label,sum)
		H -= count/sum * log(count/sum,2)
	return H

def infoGain(ds,col):
	attrs = ds[col].value_counts()
	num_attrs = attrs.sum()
	gain = 0
	print("calc info gain attribute:", col)
	for i in range(0,len(attrs)):
		label = attrs.index[i]
		count = attrs[i]
		hsv = entropy(ds[ds[col] == label][target].value_counts())
		attr_gain = (count/num_attrs)*hsv
		gain += attr_gain
		#print(label,attr_gain)
	
	Hs = entropy(ds[target].value_counts())
	#print("H(S): %f" % Hs)
	igain = Hs - gain
	#print("=>", igain)
	return igain

def ID3(examples, target, attributes = []):
	# check target
	if entropy(examples[target].value_counts()) == 0. :
		nLabel = examples[target].value_counts().index[0]
		print('Done. => Target:', examples[target].value_counts().index[0])
		return 
	if len(attributes) == 0:
		d = examples[target].value_counts().to_dict()
		# sort by frequency desc, then alphabetical asc
		mostFreqValue = sorted(d.items(), key=lambda x: (-x[1], x[0]))[0][0]
		print("No more attributes, most frequent value: ", mostFreqValue)
		return 

	# calculate information gain for remaining attributes
	a = []
	for col in attributes:
		a.append((col,infoGain(examples, col)))

	# sort by infogain, then alphabetical
	sorted(a, key=lambda x: (x[1],x[0]) )
	
	best_col = a[0][0]
	print('best attribute: =>', best_col)
	attributes.remove(best_col)
	
	# split on each value
	for val in examples[best_col].value_counts().index:
		print("\nSplit on value:", val)
		ID3(examples[examples[best_col] == val],target, attributes.copy())

ID3(data,target,[x for x in data.columns if x != target])
