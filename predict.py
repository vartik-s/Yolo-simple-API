from keras.models import load_model
import numpy as np
from yolo_utils import *
import keras.preprocessing.image as image
import tensorflow as tf
graph = tf.get_default_graph()
# Load the pretrained model
yolo=load_model("model_data/yolo.h5")

#load the class_ids
with open("model_data/classes.txt") as f:
    classes=f.readlines()
classes=[c.strip() for c in classes]

#load the anchor dimensions
with open("model_data/yolo_anchors.txt") as f:
    anchors=f.readline()
    anchors=[float(x) for x in anchors.split(',')]
    anchors=np.array(anchors).reshape(-1,2)

input_shape=yolo.layers[0].input_shape[1:]
num_classes=len(classes)
num_anchors=len(anchors)

def predict(img_path):
    img=image.load_img(img_path,target_size=input_shape)
    img=image.img_to_array(img)
    img=img/255.0
    img=np.expand_dims(img,axis=0)
    with graph.as_default():
        output=yolo.predict(img)
    boxes,scores,class_id=evaluate_yolo(output,anchors,input_shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        boxes,scores,class_id=sess.run([boxes,scores,class_id])
    predictions=[]
    for i in range(len(class_id)):
        obj={}
        obj['class_name']=classes[class_id[i]]
        obj['xmin']=str(boxes[i,1])
        obj['ymin']=str(boxes[i,0])
        obj['xmax']=str(boxes[i,3])
        obj['ymax']=str(boxes[i,2])
        obj['confidence']=str(scores[i])
        predictions.append(obj)
    return predictions
#print(predict('temp_data/c.jpeg'))