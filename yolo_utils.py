import tensorflow as tf 
import keras.backend as K 
import numpy as np

def process_outputs(y_pred,anchors):
    """
    function to process the output of the YOLO network

    =>input
    y_pred -> the output from the final layer of the yolo model 
    of the shape (batch_size,grid_h,grid_w,num_of_anchors,5*(5+number_of_classes))

    anchors -> numpy array containing the dimensions of the priors(bounding boxes)

    =>output
    pred_xy-> the x and y coordinates of the found image normalized by the image dimensions 
    of the shape (batch_size,grid_h,grid_w,num_of_anchors,2)

    pred_wh -> the width and height of the found object normalized by the image dimensions
    of the shape (batch_size,grid_h,grid_w,num_of_anchors,2)

    pred_conf -> the confidence of the box having an object
    of the shape (batch_size,grid_h,grid_w,num_of_anchors,1)

    pred_prob -> the predicted probabilities of the classes. Note that it may not be necessarily between 0 and 1
    having shape (batch_size,grid_h,grid_w,num_of_anchors,num_classes)

    """
    
    BATCH_SIZE=y_pred.shape[0]
    grid_h=y_pred.shape[1]
    grid_w=y_pred.shape[2]
    BOXES=anchors.shape[0]
    y_pred=y_pred.reshape((BATCH_SIZE,grid_h,grid_w,BOXES,-1))
    #construct a grid tensor such that grid[n_batch,i_grid_h,j_grid_w,k_box]=[j_grid,i_grid]
    x_grid=tf.cast(tf.reshape(tf.tile(tf.range(grid_w),[grid_h]),(1,grid_h,grid_w,1,1)),'float32')
    y_grid=tf.transpose(x_grid,[0,2,1,3,4])
    grid=tf.concat([x_grid,y_grid],-1)
    grid=tf.tile(grid,[BATCH_SIZE,1,1,BOXES,1])

    conv_dims=tf.Variable(y_pred.shape[1:3])
    conv_dims=K.cast(K.reshape(conv_dims,[1,1,1,1,2]),'float32')

    
    pred_xy=(tf.sigmoid(y_pred[...,:2])+grid)/conv_dims
    pred_wh=(tf.exp(y_pred[...,2:4])*np.reshape(anchors,[1,1,1,BOXES,2]))/conv_dims
    pred_conf=tf.sigmoid(y_pred[...,4:5])
    pred_classes=K.softmax(y_pred[...,5:])

    return pred_xy,pred_wh,pred_conf,pred_classes

def box_to_coordinates(box_xy,box_wh):
    """
    Converts the coordinates from box form to coordinates form (ymin,xmin,ymax,xmax)

    =>input
    box_xy -> The center coordinates of the the given bounding box normalized by the image dimensions

    box_wh -> the width and height of the predicted bounding box normalized by the image dimensions

    =>output
    boxes -> Bounding boxes in form of (ymin,xmin,ymax,xmax) which are the top left and bottom right coordinates of the box
    """
    minn=box_xy-(box_wh/2.)
    maxx=box_xy+(box_wh/2.)
    boxes= K.concatenate([minn[...,1:2],minn[...,0:1],maxx[...,1:2],maxx[...,0:1]])
    return boxes


def filter_boxes(boxes,confidence,class_prob,threshold=0.3):
    """
    Removes all those boxes which have confidence score less than the given threshold

    input=>
    boxes -> the bounding boxes predicted by YOLO
    having shape (batch_size,grid_h,grid_w,num_of_anchors,4)

    confidence -> the confidence of the box having an object
    having shape (batch_size,grid_h,grid_w,num_of_anchors,1)

    class_prob -> the probability of each individual class 
    having shape (batch_size,grid_h,grid_w,num_of_anchors,num_of_classes)

    threshold -> the cut off confidence value. All the boxes having confidence value less than threhsold are rejected

    output=>
    boxes, scores and classes of the shape (?,4), (?,) and (?,) where ? is the number of boxes selected having score greater than threshold
    """
    box_scores=confidence*class_prob
    box_classes=K.argmax(box_scores,axis=-1)
    box_class_scores=K.max(box_scores,axis=-1)
    prediction_mask=box_class_scores>=threshold

    boxes=tf.boolean_mask(boxes, prediction_mask)
    scores=tf.boolean_mask(box_class_scores,prediction_mask)
    classes=tf.boolean_mask(box_classes,prediction_mask)

    return boxes,scores,classes

def evaluate_yolo(y_pred,anchors,image_shape,max_boxes=10,score_threshold=0.3,iou_threshold=0.5):
    """
    It evaluates the output of the YOLO network

    input=>
    y_pred -> the output of the final layer of the YOLO network
    having shape (batch_size,grid_h,grid_w,num_of_anchors,5*(5+number_of_classes))

    image_shape -> the shape of the input image

    max_boxes -> the max number of boxes to be present in the final prediction

    score_threshold -> the threshold below which all the boxes are rejected (on their confidence score)

    iou_threshold -> the minimum overlap threhold (used in non max suprression)

    output=>
    boxes-> the final coordinates of the predicted boxes in (min,max) form

    scores-> the final scores of the predicted boxes

    classes-> the class_ids of the predicted boxes
    """
    yolo_outs=process_outputs(y_pred,anchors)
    box_xy,box_wh,box_conf,box_class=yolo_outs

    boxes=box_to_coordinates(box_xy,box_wh)
    boxes,scores,classes=filter_boxes(boxes,box_conf,box_class,score_threshold)
    
    height=image_shape[0]
    width=image_shape[1]
    image_ratios=tf.Variable([height,width,height,width],dtype='float32')
    image_ratios=tf.reshape(image_ratios,[1,4])
    boxes=boxes*image_ratios

    boxes=tf.cast(boxes,'float32')
    scores=tf.cast(scores,'float32')
    max_boxes=tf.Variable(max_boxes,dtype='int32')
    nms_index=tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)
    boxes=tf.gather(boxes,nms_index)
    scores=tf.gather(scores,nms_index)
    classes=tf.gather(classes,nms_index)

    return boxes,scores,classes
