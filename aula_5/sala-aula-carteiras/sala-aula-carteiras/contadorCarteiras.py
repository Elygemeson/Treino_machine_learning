import cv2
import pickle
import numpy as np

vagas = []
with open('cadeiras_vagas.pkl','rb') as arquivo:
    vagas = pickle.load(arquivo)

video = cv2.VideoCapture('cadeirasocupadas.png') #capturando o video gravado
#video = cv2.VideoCapture(0) #capturando o video em tempo real da camera

while True:
    check,img = video.read()
    imgCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgTh = cv2.adaptiveThreshold(imgCinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian = cv2.medianBlur(imgTh,5)
    kernel = np.ones((3,3),np.int8)
    imgDil = cv2.dilate(imgMedian,kernel)

    vagasAbertas = 0

    for x, y, w, h in vagas:
        vaga = imgDil[y:y+h,x:x+w]
        #cv2.rectangle(img, (x,y),(x+w,y+h),(255))
        count= cv2.countNonZero(vaga)
        cv2.putText(img,str(count),(x,y+h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        if count < 4000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255, 0), 2)
            vagasAbertas +=1
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 255), 2)            

        cv2.rectangle(img, (90,0), (415,60), (0,255,0), -1)
        cv2.putText(img, f'LIVRE: {vagasAbertas}/8', (95, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    cv2.imshow('video',img)
    cv2.waitKey(8989)