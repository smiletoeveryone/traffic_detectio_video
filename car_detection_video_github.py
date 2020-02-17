from imageai.Detection import VideoObjectDetection
import os
 
execution_path = "/path of ImageAI-master/"
 
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
os.path.join(execution_path, 'models','resnet50_coco_best_v2.1.0.h5'))
detector.loadModel()
 

custom_objects = detector.CustomObjects(car=True, bicycle=False, motorcycle=True, person=True) # you can define here what object you like to detect
video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects,input_file_path=os.path.join(execution_path, 'videos', 'traffic_detection.mp4'),output_file_path=os.path.join(execution_path, 'videos','traffic_custom_detected'),frames_per_second=20,log_progress=True)
print(video_path)
