#!/usr/bin/env python
# coding: utf-8

# In[29]:


#tensorflow and keras classification 

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from imutils.object_detection import non_max_suppression
from kunalsearch.pyramid_slider import sliding_window
from kunalsearch.pyramid_slider import image_pyramid
import argparse
import cv2
import imutils
import time
import numpy as np

#parse the argument 

ap = argparse.ArgumentParser()
ap.add_argument("-i" , "--image" , required=True, help ="path to the image directory")
ap.add_argument("-s" ,"--size" , type= str,  default= "(200,150)", help ="ROI size in pixels")
ap.add_argument("-c" ,"--min-conf" ,type =float , default=0.9, help ="min prob for filtering weak detections")
ap.add_argument("-v" ,"--visualize" ,type =float , default =-1 , help = "extra visualize for debugging")
args= vars(ap.parse_args())



WIDTH = 700
PYR_SCALE =1.5
WIN_STEP =16
ROI_SIZE =eval(args["size"])
INPUT_SIZE =(224,224)

print("[INFO] load our network....")

model =ResNet50(weights ="imagenet" , include_top =True)

orig =cv2.imread(args["image"])
orig =imutils.resize(orig, width =WIDTH)

(H,W) =orig.shape[:2]


pyramid =image_pyramid(orig , scale =ROI_SIZE , minSize =ROI_SIZE)


rois=[]
locs=[]

start =time.time()



#loop over the pyramid

for image in pyramid:
    scale = W / float(image.shape[1])
    
    for (x,y ,roiOrig) in sliding_window(image , WIN_STEP, ROI_SIZE):
        x= int(x*scale)
        y=int(y*scale)
        w =int(ROI_SIZE[0]*scale)
        h= int(ROI_SIZE[1]*scale)
        
        
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        
        rois.append(roi)
        locs.append((x,y,x+w ,y+h))
        
        
        
        #checking for visualisation
        
        if args["visualize"]>0:
            
            clone =orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
            
        
    
            cv2.imshow("Visualize", clone)
            cv2.imshow("ROI" , roi0rig)
            cv2.waitKey(0)
            

end= time.time()
print("loading the info".format(end-start))


rois =np.array(rois , dtype ="float32")

print("[Info] ROI is claasifying")
start = time.time()
preds = model.predicts(rois)
end =time.time()


print("[INFO] classification is done".format(end-start))

preds =imagenet_utils.decode_predictions(preds , top=1)
label ={}


for (i ,p) in enumerate(preds):
    (imagenetID, label ,prob) =p[0]
    
    
    if prob >= args["min_conf"]:
        box =locs[i]
        
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L
    
    

for label in labels.keys():
    print("showing info...".format(label))
    
    clone= orig.copy()
    
    for (box,probs) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
      

    cv2.imshow("Before", clone)
    clone =orig.copy()
    
    
    boxes= np.array(p[0] for p  in labels[label])
    proba =np.array(p[1] for p in labels[label])
    boxes = non_max_suppression(boxes ,proba)
    
    
    for(startX , startY,endX ,endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("After", clone)
    cv2.waitKey(0)






















# In[ ]:




