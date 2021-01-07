import numpy as np
import cv2
import imutils
import urllib.request

prototxt="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
confthresh=0.2 #to check confidence level ie only if object other than backgrd is present, it should process

classes=["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","pen"]

#random colors for random objects in image
colors = np.random.uniform(0, 255, size=(len(classes), 3))


print("*****loading model*****")
net=cv2.dnn.readNetFromCaffe(prototxt,model)
print("*****model loaded*****")
print("*****starting camera feed*****")
cam=cv2.VideoCapture(0)
#time.sleep(2.0)
url=' http://192.168.1.3:8080/shot.jpg'

while True:

    '''imgresp=urllib.request.urlopen(url) #get image from webcam
    imgnp=np.array(bytearray(imgresp.read()),dtype=np.uint8) #convert into array
    img=cv2.imdecode(imgnp,-1)
    frame=imutils.resize(img,width=500)'''

    _,frame=cam.read()
    frame=imutils.resize(frame,width=500)


    (h,w)=frame.shape[:2]
    #ssd pre reqisite is image of 300x300 so we ar resizing again
    imgresizeblob=cv2.resize(frame,(300,300))
    blob=cv2.dnn.blobFromImage(imgresizeblob,0.007843,(300,300),127.5)

    net.setInput(blob)
    detections=net.forward() #proceed for classificcation
    detshape=detections.shape[2]
    for i in np.arange(0,detshape):
        confidence = detections[0, 0, i, 2]
        if confidence > confthresh:
            idx = int(detections[0, 0, i, 1])
            if(idx==5.0):
                label="I need water"
            else:
                label = "{}: {:.2f}%".format(classes[idx],confidence * 100)    
            print("ClassID:",detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #label = "{}: {:.2f}%".format(classes[idx],confidence * 100)
            label="I need water"

            cv2.rectangle(frame, (startX, startY), (endX, endY),colors[idx], 2)
            if startY - 15 > 15:
                y = startY - 15
            else:
                y=startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    cv2.imshow("Frame", frame)
    if(cv2.waitKey(1) & 0xFF == ord('x')):
        break
cam.release()
cv2.destroyAllWindows()
