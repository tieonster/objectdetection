import os

# Before running this file, split video up into frames using process.py first
# Make sure images are in jpg format and are named 1.jpg, 2.jpg, 3.jpg, etc...

# directory where images are stored
directory = r'/content/drive/My Drive/XXX'

for root, dirs, files in os.walk(directory):
    number_of_images = len(files)
    
# Set count to 0 so that can loop through images
count = 0
for i in range(1,number_of_images+1):
      input_w, input_h = 416, 416
      photo_file_name = os.path.join(directory, str(i) + '.jpg') #prints out location of image

      image, image_w, image_h = load_image_pixels(photo_file_name, (input_w,input_h))

      netout = model.predict(image)
      
      # Pre-defined anchors
      anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
      class_threshold = 0.6

      # returns boxes
      boxes = []
      for i in range (len(netout)):
          boxes += decode_netout(netout[i][0], anchors[i], class_threshold, input_h, input_w)

      correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

      do_nms(boxes, 0.5)

      # Predefined labels
      labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


      v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

      # for i in range(len(v_boxes)):
      #     print(v_labels[i], v_scores[i])

      draw_boxes(photo_file_name, v_boxes, v_labels, v_scores)
      count += 1
      # Folder to save images in
      pyplot.savefig('/content/drive/My Drive/XXX/test' + str(count) + '.jpg')
      pyplot.clf()
      print('saved ' + str(count))
