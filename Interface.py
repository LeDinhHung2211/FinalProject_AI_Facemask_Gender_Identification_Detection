
from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceMask_model = load_model('faceMask.h5')  
gender_model = load_model('gender_final.h5')    
id_model = load_model('id_ver13.h5')

gender_labels = ['Male', 'Male', 'Male', 'Male']
faceMask_labels = ['No Mask', 'Mask']
id_labels = ['BaHuy','BaHuy','DinhHung','DinhHung','MinhTuan','MinhTuan','QuangHuy','QuangHuy'] 



class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(950,800)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 60, 901, 561))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(370, 650, 200, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "START"))
        self.pushButton.clicked.connect(self.use_camera)

    def use_camera(self, Dialog):
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.is_running = True
    
    def ImageUpdateSlot(self, Image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QtCore.QThread):
    ImageUpdate = QtCore.pyqtSignal(QtGui.QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            # frame = cv2.resize(frame,[1100,720])
            # frame = cv2.resize(frame,[1240,720])
            # frame = cv2.resize(frame,[500,800])
            labels=[]

 
            faces = face_classifier.detectMultiScale(frame,1.3,5)

            for (x,y,w,h) in faces:
                roi_color=frame[y:y+h,x-15:x+w+15]
                
                roi_color=cv2.resize(roi_color,(150,150),interpolation=cv2.INTER_AREA)
                
                #Get image ready for prediction
                roi=roi_color.astype('float')/255.0  #Scale
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)  
                #Gender
                gender_predict = gender_model.predict(np.array(roi_color).reshape(1,150,150,3))
                preds=gender_model.predict(roi)[0] 
                label=gender_labels[preds.argmax()]  #Find the label
                label_position=(x,y-40)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
                #FaceMask
                faceMask_predict = faceMask_model.predict(np.array(roi_color).reshape(1,150,150,3))
                faceMask_predict = (faceMask_predict>= 0.5).astype(int)[:,0]
                faceMask_label=faceMask_labels[faceMask_predict[0]] 
                faceMask_label_position=(x,y+h+60) 
                cv2.putText(frame,faceMask_label,faceMask_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                #face_id
                id_predict = id_model.predict(roi)[0] 
                # id_predict = (id_predict>= 1).astype(int)[:,0]
                id_label=id_labels[id_predict.argmax()]
                id_label_position=(x+200,y+h+60) #50 pixels below to move the label outside the face
                cv2.putText(frame,id_label,id_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
                cv2.rectangle(frame,(x-15,y-20),(x+w+15,y+h+20),(50,50,255),2)                
            # cv2.imshow('hunhun', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QtGui.QImage(Image.data, Image.shape[1], Image.shape[0], QtGui.QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

        # listen to press F5 to stop thread
        if self.ThreadActive:
            self.ThreadActive = False
            Capture.release()
            cv2.destroyAllWindows()

        
    def stop(self):
        self.ThreadActive = False
        self.quit()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
