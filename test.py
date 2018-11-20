
import pandas as pd
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
	print("Info Gain Column:", col)
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


def ID3(ds, processedCols = []):
	# check target
	if entropy(ds[target].value_counts()) == 0. :
		print('Done. => Target:', ds[target].value_counts().index[0])
		return

	# calculate information gain for remaining columns
	a = []
	for col in ds.columns:
		if col == target or col in processedCols:
			continue
		a.append((col,infoGain(ds, col)))

	if len(a) == 0:
		print("No more columns left to process")
		return
	
	# sort by infogain, then alphabetical
	sorted(a, key=lambda x: (x[1],x[0]) )
	
	best_col = a[0][0]
	print('best column: =>', best_col)
	processedCols.append(best_col)

	# split on each attribute
	for attr in ds[best_col].value_counts().index:
		print("\nSplit on attribute:", attr)
		ID3(ds[ds[best_col] == attr],processedCols.copy())


ID3(data)

