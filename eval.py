import os

from dataset import load_batch

def eval_model(recognizer, meta_file):
    if not os.path.exists(meta_file):
        print(meta_file, "doesn't exits!")

    [list_img, list_lable], num_sample=load_batch(meta_file) 

    if num_sample<1:
        return 0

    check=0
    for i in range(num_sample):
        predection, _ =recognizer.predict(list_img[i])
        check+=predection==list_lable[i]
    return float(check)/num_sample