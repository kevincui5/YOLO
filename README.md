In this exercise I'd like to implement bounding box filtering part of the powerful
YOLO model for object detection problem, specificly, car detection, using tensorflow.
Images are passed through CNN, and the resulting output, containing confidence 
probability that there's some object, bounding box coordinates, detection probabilities
for each of the 80 object classes.
The above input is taken by two filtering functions I implemented to give the final output.
The first function, yolo_filter_boxes, filters bounding box by a specified threshold
 on class scores.  The second function, yolo_non_max_suppression, uses non max 
 suppression algorithm.  Also implemented IOU but not used in the program, only 
 for exercise.
 Detail information on the format of input images, the YOLO encoding architecture used
 for this exercise, how test image is rescaled etc, please check out the programming
 exercised description for coursera CNN course weeks 3.
The CNN model is implemented by Allan Zelener in YAD2K with Keras.
Because of the file size limit by github, yolo.h5 is not included in this project.

Reference:
Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)
Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)
Allan Zelener - YAD2K: Yet Another Darknet 2 Keras