# objectdetection
Object Detection Project from videos with Yolov3 

Referenced from the following project:
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

In this project, we will detect objects from videos and draw bounding boxes around them to label them.

We will use the pre-trained weights from the yolov3 model that can be found here: 
https://pjreddie.com/media/files/yolov3.weights

We will split the video into individual frames (images) and run the model through each image to get the bounding boxes using python's cv2 library.
After getting the bounding boxes around the objects, we will combine all the images back together to get the video.

Here is an outline of files:

model.py -- Code that creates the yolov3 keras model and loads pre-trained weights into model

functions.py - Includes a list of functions that is needed to process outputs from model and load image files

main.py - Code to run model on image files stored in a directory

process.py - Code to convert video to images and images to video


After running main.py, you will get a directory of images that have bounding boxes around them.
Run the folder of images using process.py to get the video! :)

However, the accuracy of the yolov3 model is not fantastic and to acheive better results, you should further train the yolo model on a larger dataset.
