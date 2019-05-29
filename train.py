import numpy as np 
import datetime
import os
import cv2
import sys

import config 
from dataset import load_batch
from init_dataset_meta import init_dataset_meta
from eval import eval_model

def train(meta_files):
   
    if not os.path.exists(config.OUTPUT_DIR):
        os.mkdir(config.OUTPUT_DIR)

    
    #recognizer=cv2.face.EigenFaceRecognizer_create()
    #recognizer=cv2.face.FisherFaceRecognizer_create()
    recognizer=cv2.face.LBPHFaceRecognizer_create()

    global_accuracy=0
    print('\nProcessing...')

    [list_img, list_label], num_sample=load_batch(meta_files[0])
    if num_sample<1:
        return

    recognizer.train(list_img, np.array(list_label))
    train_accuracy=eval_model(recognizer, meta_files[0])
    val_accuracy=eval_model(recognizer, meta_files[1])
    test_accuracy=eval_model(recognizer, meta_files[2])

    print('\nRESULT: Number of images: %d, Train accuracy: %g, Test accuracy: %g' %(
        num_sample,
        train_accuracy,
        test_accuracy
    ))

    recognizer.save(os.path.join(config.OUTPUT_DIR,config.OUTPUT_MODEL_FILE))

if __name__=='__main__':
    if len(sys.argv)==4:
        train(sys.argv[1:])
    else:
        train([])
        
