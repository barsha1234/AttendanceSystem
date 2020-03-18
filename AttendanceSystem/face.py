import cv2
import os
import PIL.Image
import numpy as np
from pandas.io import sql
import pandas as pd
import datetime
import time
import pymysql
import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox


connection=pymysql.connect('localhost','root','','project')

#connection=sqlalchemy.create_engine('mysql+pymysql://root:@localhost/project')



face_classifier = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')

def capturephoto(rollno,name,email,department,arrival,gender):
    if (rollno !=None and name !=None):
        count=0
        video= cv2.VideoCapture(0)
        while True:
            check,frame=video.read()
            if check==False:
                continue
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face=face_classifier.detectMultiScale(grey,1.3,5)
            for x,y,w,h in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                count=count+1
                cv2.putText(frame,str(count),(x,y+h),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                cv2.imwrite('trainingimages/'+name+'.'+rollno+'.'+str(count)+'.jpg',grey[x:x+w,y:y+h])
                cv2.imshow('taking picture',frame)
                if count==100:
                    video.release()
                    cv2.destroyAllWindows()
                    col=['userid','name','department','email','stoa','gender']
                    df=pd.DataFrame(columns=col)
                    df.loc[len(df)]=[int(rollno),name,department,email,arrival,gender]
                    df.to_sql(con=connection,name='userdetails',if_exists='append',index=False)
                    return 'All images collected'
                    break
    
def TrainImage():
    path='trainingimages'
    imagePaths = [os.path.join(path,f)for f in os.listdir(path)]
    face,rollno = [],[]
    for imagePath in imagePaths:
        pilImage = PIL.Image.open(imagePath).convert('L')
        imageNP = np.array(pilImage,'uint8')
        roll = int(os.path.split(imagePath)[-1].split('.')[1])
        face.append(imageNP)
        rollno.append(roll)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face, np.array(rollno))
    recognizer.save('TrainingData.yml')
    return 'Images Trained Sucessfully'


def TrackFace():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('TrainingData.yml')
    xy=pd.read_sql('SELECT userid,name FROM userdetails',con=connection)
    cam=cv2.VideoCapture(0)
    col_names=['userid','date','time','name']
    attendance=pd.DataFrame(columns=col_names)
    font=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        a,img=cam.read()
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(grey,1.2,5)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            Id,conf=recognizer.predict(grey[y:y+h,x:x+w])
            if conf<50:
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timestamp=datetime.datetime.fromtimestamp(ts).strftime('%H-:%M:%S')
                nm=xy.loc[xy['userid']==Id]['name'].values
                text=str(Id)+'-'+nm
                attendance.loc[len(attendance)]=[Id,nm[0],date,timestamp]
            else:
                 Id='unknown'
                 text=str(Id)
            if conf>75:
                countfile=len(os.listdir('unknownphoto'))+1
                cv2.imwrite('unknownphoto\photo'+str(countfile)+'.jpg',img[y:y+h,x:x+w])
            cv2.putText(img,str(text),(x,y+h),font,1,(0,0,255),2)
        attendance=attendance.drop_duplicates(subset=['userid'],keep='first')
        cv2.imshow('show ur face',img)
        if cv2.waitKey(1)==13:
            break
    cam.release()
    cv2.destroyAllWindows()
    attendance.to_sql(con=connection,name='attendance',if_exists='append',index=False)
    if attendance.empty:
        return 'no attendance'
    else:
        return 'attendance taken'





#screen design part --------------------------------------------------------------------------------------------------------------------------------------------------


        
window=tkinter.Tk()
window.state('zoomed')
window.title('FaceRecognitionAttendanceSystem')
window.configure(bg='#ffc266')
tkinter.Label(window, text='FACE RECOGNITION BASED ATTENDANCE SYSTEM',bg='#ffc266', font=('Arial_Black',30),pady=20).place(x=140,y=0)

gender=tkinter.StringVar()




top_frame= tkinter.LabelFrame(window,text='Search',font=('Times New Roman',30),bg='#ffc266')
top_frame.place(x=98,y=80)

def clear1():
    ids.delete(0,'end')
    mail.delete(0,'end')

tkinter.Label(top_frame, text='Id:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=0,padx=5,pady=10)
ids=tkinter.Entry(top_frame, width=5)
ids.grid(row=0, column=1,padx=5,pady=10)
tkinter.Label(top_frame, text='Name:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=0,column=3,padx=14,pady=10)
mail=tkinter.Entry(top_frame, width=30)
mail.grid(row=0, column=4,padx=3,pady=10,columnspan=3)
a=tkinter.Button(top_frame, text='FIND', fg='black', bg='#EADB93',font=('Times New Roman Baltic',10)).grid(row=4, column=5,padx=10,pady=10)
b=tkinter.Button(top_frame, text='CLEAR', fg='black', bg='#EADB93',font=('Times New Roman Baltic',10),command=clear1).grid(row=4, column=6,padx=10,pady=10)




top_frame= tkinter.LabelFrame(window,text='Add a new record',font=('Times New Roman',30),bg='#ffc266')
top_frame.place(x=100,y=250)




def clear2():
    roll.delete(0,'end')
    name.delete(0,'end')
    mail2.delete(0,'end')
    dept.delete(0,'end')
    arv.delete(0,'end')

def addimage():
    ROLL= roll.get()
    NAME=name.get()
    MAIL2=mail2.get()
    DEPT=dept.get()
    ARV=arv.get()
    GENDER=gender.get()
    z=capturephoto(ROLL,NAME,MAIL2,DEPT,ARV,GENDER)
    messagebox.showinfo('Notification',z)


tkinter.Label(top_frame, text='Enter RollNo:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=0,padx=5,pady=10)
roll=tkinter.Entry(top_frame, width=15)
roll.grid(row=0, column=1,padx=5,pady=10)

tkinter.Label(top_frame, text='Enter Name:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=0,column=2,padx=5,pady=10)
name=tkinter.Entry(top_frame, width=20)
name.grid(row=0, column=3,padx=5,pady=10)

tkinter.Label(top_frame, text='Enter Email Id:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=2,padx=5,pady=10)
mail2=tkinter.Entry(top_frame, width=20)
mail2.grid(row=2, column=1,padx=5,pady=10)

tkinter.Label(top_frame, text='Enter Department:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=2,column=2,padx=5,pady=10)
dept=tkinter.Entry(top_frame, width=20)
dept.grid(row=2, column=3,padx=5,pady=10)

tkinter.Label(top_frame, text='Enter Arrival Time:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=3,padx=5,pady=10)
arv=tkinter.Entry(top_frame, width=10)
arv.grid(row=3, column=1,padx=5,pady=10)

tkinter.Label(top_frame, text='Enter Gender:' ,font=('Arial_Black',15),bg='#ffc266').grid(row=4,padx=5,pady=10)
rad1=Radiobutton(top_frame,text='male',value='M',variable=gender)
rad1.grid(row=4,column=1)
rad2=Radiobutton(top_frame,text='female',value='F',variable=gender)
rad2.grid(row=4,column=2)

x=tkinter.Button(top_frame, text='TAKE IMAGE', fg='black', bg='#EADB93',font=('Times New Roman Baltic',10),command=addimage).grid(row=5, column=4,padx=10,pady=10)
y=tkinter.Button(top_frame, text='CLEAR', fg='black', bg='#EADB93',font=('Times New Roman Baltic',10),command=clear2).grid(row=5, column=5,padx=20,pady=10)



bottom_frame= tkinter.LabelFrame(window,bg='#ffc266')
bottom_frame.place(x=300,y=600)

def close():
    window.destroy()


def Trainbtn():
     s=TrainImage()
     messagebox.showinfo('Notification',s)
    
def Takeattndnce():
    p=TrackFace()
    messagebox.showinfo('Notification',p)
    
    
p=tkinter.Button(bottom_frame, text='Train Image', fg='black', bg='#EADB93',font=('Times New Roman Baltic',20),command=Trainbtn).grid(row=0, column=1,padx=10,pady=10)
q=tkinter.Button(bottom_frame, text='Take Attendance', fg='black', bg='#EADB93',font=('Times New Roman Baltic',20),command=Takeattndnce).grid(row=0, column=2,padx=10,pady=10)
r=tkinter.Button(bottom_frame, text='Daily Report', fg='black', bg='#EADB93',font=('Times New Roman Baltic',20)).grid(row=0, column=3,padx=10,pady=10)
s=tkinter.Button(bottom_frame, text='Close', fg='black', bg='#EADB93',font=('Times New Roman Baltic',20),command=close).grid(row=0, column=4,padx=10,pady=10)





window.mainloop()






                          

    
    

   
                


