import cv2
import face_recognition as fr
import os

encoders = []
nomes = []

def criarEncoders():
    lista = os.listdir('Pessoas')

    for arquivo in lista:
        inAtual = fr.load_image_file(f'Pessoas/{arquivo}')
        inAtual = cv2.cvtColor(inAtual,cv2.COLOR_BGR2RGB)
        encoders.append(fr.face_encodings(inAtual)[0])
        nomes.append(os.path.splitext(arquivo)[0])

def compararWEbcam():
    video = cv2.VideoCapture(0)

    while True:
        check, img = video.read()
        imP = cv2.resize(img,(0,0),None,0.25,0.25)
        imgP = cv2.cvtColor(imgP,cv2.COLOR_BGR2RGB)

        try:
            faceLoc = fr.face_locations(imgP)[0]
        except:
            faceLoc = []


        if faceLoc:
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1+4,x2+4,y2+4,x1+4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            encodeImg = fr.face_encodings(imgP)[0]

            for id,enc in enumerate(encoders):
                comp = fr.compare_faces([encodeImg],enc)
                if comp[0]:
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),-1)
                    cv2.putText(img,nomes[id],(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    print(nomes[id])


        cv2.imshow('Webcam',img)
        cv2.waitKey(1)

criarEncoders()
compararWEbcam()
#print(nomes)



