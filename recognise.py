from dataset import load_img
import os
import cv2
import config
import sys
import numpy

def recognise(img_file, uid_file):
    if not os.path.exists(img_file) or not os.path.exists(uid_file):
        print('ERROR')
        return
    
    #recognizer=cv2.face.EigenFaceRecognizer_create()
    #recognizer=cv2.face.FisherFaceRecognizer_create()
    recognizer=cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(os.path.join(config.OUTPUT_DIR, config.OUTPUT_MODEL_FILE))

    face_cascade = cv2.CascadeClassifier('cvdata\haarcascade_frontalface_default.xml')

    img=cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces)==0:
        print('No one found!')
        return

    print('\nFile name: ', img_file)
    print('ID(s): ', end='')

    for k in range(len(faces)):
        face = numpy.array([], dtype = 'uint8')
        x, y, h, w = faces[k]
        face.resize((h, w, 3))
        for i in range(w):
            for j in range(h):
                face[i][j] = img[y + i][x + j]
    
        file_name, _=img_file.split('.')
        new_file=file_name+str(k)+'.jpg'
        cv2.imwrite(new_file, face)
        
    for k in range(len(faces)):
        file_name, _=img_file.split('.')
        new_file=file_name+str(k)+'.jpg'
        img=load_img(new_file)
        dir_path = os.path.dirname(img_file)

        
        prediction, score=recognizer.predict(img)
        if score<0:
            print('unknown')
        else:
            with open(uid_file) as f:
                uids=[line.strip() for line in f.readlines()]
                if len(uids)<=prediction:
                    print('unknown')
                else:
                    os.rename(new_file, dir_path+'\\'+uids[prediction]+'.jpg')
                    print(uids[prediction], end=' ')
    print('\n')
              

if __name__=='__main__':
    import glob
    files=glob.glob(sys.argv[1]+'/*')
    for img in files:
        recognise(img, 'uid.txt')

