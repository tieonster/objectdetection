import cv2
import os

image_folder = "C:/Users/Shaun/Desktop/XXX"
video_name = 'video.avi'

for root, dirs, files in os.walk(image_folder):
    number_of_images = len(files)

images = []
for i in range(1,number_of_images+1):
    images.append("test" + str(i) + ".jpg")

print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 4, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
