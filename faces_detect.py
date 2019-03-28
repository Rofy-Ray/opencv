
# coding: utf-8

# In[41]:


# import packages
import numpy as np
import argparse
import cv2


# In[42]:


# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())


# In[43]:


# load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# In[37]:


# load input image & construct input blob for image by resizing to fixed 300x300 px & normalize
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


# In[38]:


# pass blob through network & obtain detections & predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()


# In[40]:


# loop over detections
for i in range(0, detections.shape[2]):
#     extract confidence(probability) associated with predictions
    confidence = detections[0, 0, i, 2]
#     filter out weak detections by ensuring confidence is greater than minimum confidence
    if confidence > args["confidence"]:
#         compute (x,y) coordinates for bounding box for object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        box = box.astype("int")
        startX, startY, endX, endY = (box[0], box[1], box[2], box[3])
#     draw bounding box of face with associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
# show output image
ims = cv2.resize(image, (960, 540))
cv2.imshow("Output", ims)
cv2.waitKey(0)

