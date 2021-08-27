import cv2 #Importing OpenCV library (installation : file,settings,project,add,search open cv)

#img = cv2.imread('lena.PNG') #(initial for testing without opening webcam)
thresh = 0.5 #Margin of error / be atleast this much sure before declaring name to obj / can be changed later for accuracy
cap = cv2.VideoCapture(0) #Open webcam/camera / 0th camera active
if not cap.isOpened(): #for case of non-working of camera
    print("Cannot open camera")
    exit()
cap.set(3,640)#set limits to cameras (Defaults)
cap.set(4,480)


classNames =[] #Assign names of respective objects / available on GIT
classFile = 'coco.names' #import names from folder
with open(classFile,'rt') as f : #Converting TXT file to a list in python
        classNames = f.read().rstrip('\n').rsplit('\n') #Strip txt file data into list according to lines
#print(classNames)#not needed / to check if txt file is imported as list or not


config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #Data for conversion of img data to python values
weights = 'frozen_inference_graph.pb' #comparing img data to other python value for recognition

#net = cv2.dnn_DetectionModel(weights,config) activate detect mode of opencv library
net = cv2.dnn_DetectionModel(weights , config)
net.setInputSize(320,320)#Reconfigure dimensions of the inputs of webcam imgs
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True : #for multiple continous img input from webcam for recognition (video = cascading multiple images)
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thresh)#refrencing id of name , confidence/accuracy/box around img to the image thresh
        print(classIds,bbox)#forming a box around the live/still image
        if len(classIds) !=0:#even if at one point the image does not match , the scannng should not stop
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):#using flatten function for non-overlapping images/objects
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)#dimensions and color of the rectangle
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),#putting name inside the box for identification
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img, str(round(confidence*100,2)), (box[0] + 250, box[1] + 30),#putting confidence/thresh/accuracy in box
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO",img)#show image/video as output
        cv2.waitKey(1)#delay command to keep img window from closing