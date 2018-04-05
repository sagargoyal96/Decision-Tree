from __future__ import print_function
import time,sys,statistics,csv
import numpy as np

## The possible attributes in the data with the prediction at index 0. Smaller names for brevity.
attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]

## Get the encoding of the csv file by replacing each categorical attribute value by its index.
wc_l = "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(", ")
edu_l = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(", ")
mar_l = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(", ")
occ_l = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(", ")
rel_l = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(", ")
race_l = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(", ")
sex_l = "Female, Male".split(", ")
nc_l = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(", ")
encode = {
    "rich"   : {"0":0,"1":1},
    "wc"     : {wc_l[i]:i for i in range(len(wc_l))},
    "edu"    : {edu_l[i]:i for i in range(len(edu_l))},
    "mar"    : {mar_l[i]:i for i in range(len(mar_l))},
    "occ"    : {occ_l[i]:i for i in range(len(occ_l))},
    "rel"    : {rel_l[i]:i for i in range(len(rel_l))},
    "race"   : {race_l[i]:i for i in range(len(race_l))},
    "sex"    : {sex_l[i]:i for i in range(len(sex_l))},
    "nc"     : {nc_l[i]:i for i in range(len(nc_l))},
    }

attr_sizes=[2,2,len(wc_l),2,len(edu_l),2,len(mar_l),len(occ_l),len(rel_l),len(race_l),len(sex_l),2,2,2,len(nc_l)]


def medians(file):
    """
    Given a csv file, find the medians of the categorical attributes for the whole data.
    params(1): 
        file : string : the name of the file
    outputs(6):
        median values for the categorical columns
    """
    fin = open(file,"r")
    reader = csv.reader(fin)
    age, fnlwgt, edun, capg, capl, hpw = ([] for i in range(6))
    total = 0
    for row in reader:
        total+=1
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        # print("l= ",l)
        age.append(int(l[0]));
        fnlwgt.append(int(l[2]));
        edun.append(int(l[4]));
        capg.append(int(l[10]));
        capl.append(int(l[11]));
        hpw.append(int(l[12]));
    fin.close()
    return(statistics.median(age),statistics.median(fnlwgt),statistics.median(edun),statistics.median(capg),statistics.median(capl),statistics.median(hpw))

def preprocess(file):
    """
    Given a file, read its data by encoding categorical attributes and binarising continuos attributes based on median.
    params(1): 
        file : string : the name of the file
    outputs(6):
        2D numpy array with the data
    """
    # Calculate the medians
    agem,fnlwgtm,edunm,capgm,caplm,hpwm = medians(file)
    fin = open(file,"r")
    reader = csv.reader(fin)
    data = []
    total = 0
    for row in reader:
        total+=1
        # Skip line 0 in the file
        if(total==1):
            continue

        l = [x.lstrip().rstrip() for x in row]
        t = [0 for i in range(15)]
        
        # Encode the categorical attributes
        t[0] = encode["rich"][l[-1]]; t[2] = encode["wc"][l[1]]; t[4] = encode["edu"][l[3]]
        t[6] = encode["mar"][l[5]]; t[7] = encode["occ"][l[6]]; t[8] = encode["rel"][l[7]]
        t[9] = encode["race"][l[8]]; t[10] = encode["sex"][l[9]]; t[14] = encode["nc"][l[13]]
        
        # Binarize the numerical attributes based on median.
        # Modify this section to read the file in part c where you split the continuos attributes baed on dynamic median values.
        t[1] = float(l[0])>=agem; t[3] = float(l[2])>=fnlwgtm; t[5] = float(l[4])>=edunm;
        t[11] = float(l[10])>=capgm; t[12] = float(l[11])>=caplm; t[13] = float(l[12])>=hpwm;
        
        # Convert some of the booleans to ints
        data.append([int(x) for x in t])
    
    return np.array(data,dtype=np.int64)

## Read the data
train_data = preprocess("train.csv")

valid_data = preprocess("valid.csv")
test_data = preprocess("test.csv")
# print(train_data)
# print("The sizes are ","Train:",len(train_data),", Validation:",len(valid_data),", Test:",len(test_data))

class node:
	# global node_counter
	def __init__(self, data, xj, childlist):
		global node_counter
		self.split_attr=xj
		self.childlist=childlist
		self.data=data
		self.leaf=0
		self.leaf_value=0
		node_counter+=1


	def setleaf(self, value):
		self.leaf=1
		self.leaf_value=value


node_counter=0

def maketree(data):
	global node_counter
	flag0=0
	flag1=0
	for item in data:
		if item[0]==0:
			flag0=1
		if item[0]==1:
			flag1=1

	if flag0==0:
		leaf=node(data,0, [])
		# node_counter+=1
		leaf.setleaf(1)
		return leaf

	if flag1==0:
		leaf=node(data,0,[])
		# node_counter+=1
		leaf.setleaf(0)
		return leaf

	attr, info_g=choose_best(data)
	if info_g==0:
		# find majority class
		leaf=node(data,0,[])

		count0=0
		count1=0
		for item in data:
			if item[0]==0:
				count0+=1
			else:
				count1+=1
		if count0>=count1:
			leaf.setleaf(0)
		else:
			leaf.setleaf(1)

		return leaf


	no_examples=len(data)
	types=attr_sizes[attr]
	sep_list=[]
	for i in range(types):
		temp_list=[]
		for j in range(no_examples):
			if data[j][attr]==i:
				temp_list.append(data[j])
		sep_list.append(temp_list)

	node_list=[]
	for item in sep_list:
		node_list.append(maketree(item))

	my_node=node(data, attr, node_list)
	# node_counter+=1
	return my_node




def find_hy_s(sep_list):
	answers=[]
	for item in sep_list:
		# print(len(sep_list))
		total_y=0
		total_y1=0
		total_y0=0
		for elem in item:
			if elem[0]==0:
				total_y0+=1
				total_y+=1
			else:
				total_y1+=1
				total_y+=1
		# print(total_y)
		if total_y==0:
			answers.append(0)
		else:
			Py1=total_y1/total_y
			Py0=total_y0/total_y
			hy1=0
			hy0=0
			if Py1==0:
				hy1=0
			if Py0==0:
				hy0=0
			if Py1!=0 and Py0 !=0:
				hy1=Py1*((-1)*np.log2(Py1))
				hy0=Py0*((-1)*np.log2(Py0))
			answers.append(hy1+hy0)
	return answers

def make_list_of0(length):
	lis=[]
	for i in range(length):
		lis.append(0.0)
	return lis

def inf_gain(data, attr):
	no_examples=len(data)
	types=attr_sizes[attr]
	sep_list=[]
	prob_x=make_list_of0(types)
	for i in range(types):
		temp_list=[]
		for j in range(no_examples):
			if data[j][attr]==i:
				temp_list.append(data[j])
		sep_list.append(temp_list)
		prob_x[i]+=len(temp_list)/no_examples
		# print(len(temp_list))

	hy_s=find_hy_s(sep_list)
	# print("hys= ",hy_s)
	# print("prob= ", prob_x)
	hy_givenx=0
	for i in range(types):
		hy_givenx+=prob_x[i]*hy_s[i]
	return hy_givenx


def choose_best(data):
	maxind=1

	if len(data)==0:
		print(node_counter)
	Hy=find_hy_s([data])[0]
	info_g=Hy-inf_gain(data,1)

	for i in range(2,15):
		temp_in_g=inf_gain(data,i)
		if(Hy-temp_in_g>info_g):
			maxind=i
			info_g=Hy-temp_in_g	

	# print(maxi/nd, info_g)
	return maxind, info_g

def predict_val(inp,root):
	if root.leaf==1:
		return root.leaf_value
	else:
		return predict_val(inp,root.childlist[inp[root.split_attr]])

def find_accuracy(data, root):
	correct=0
	total=len(data)
	for item in data:
		if item[0]==predict_val(item,root):
			correct+=1

	print(correct/total)



fin=maketree(train_data)
print(node_counter)
find_accuracy(test_data, fin)

































































