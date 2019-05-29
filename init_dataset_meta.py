#init_dataset_meta.py
import glob
import random
import os
import numpy as np
import sys
 
import config
 
def init_dataset_meta(data_dir = config.DATASET_DIR,
					ptrain = config.PTRAIN, pvalidate = config.PVAL, ptest = config.PTEST):
	train = []
	validate = []
	test = []
	folders = glob.glob(data_dir+'/*')
	folders.sort()

	f_uid = open('uid.txt','w')
	f_train = open('train.txt','w')
	f_val = open('validate.txt','w')
	f_test = open('test.txt','w')
	
	for (i) in folders:
		f_uid.write(i.split(os.sep)[-1] + '\n')
		dir_img = glob.glob(i+'/*')
		random.shuffle(dir_img)
		
		n = int(len(dir_img))
		ntrain = int(n*ptrain)
		nval = int(n*pvalidate)
		
		train_set = dir_img[: ntrain]
		validate_set = dir_img[ntrain : ntrain+nval]
		test_set = dir_img[ntrain+nval :]
		
		for (a) in train_set:
			f_train.write(os.path.abspath(a) + '\t' + str(folders.index(i)) + '\n')
		for (a) in validate_set:
			f_val.write(os.path.abspath(a) + '\t' + str(folders.index(i)) + '\n')
		for (a) in test_set:
			f_test.write(os.path.abspath(a) + '\t' + str(folders.index(i)) + '\n')

	f_uid.close()
	f_train.close()
	f_val.close()
	f_test.close()
	return 
if __name__ == '__main__':
	init_dataset_meta()