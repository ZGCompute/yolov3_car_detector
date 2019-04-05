Car_Detector
===========

This repository contains code for detecting parked cars in selected parking spaces from a range of video imagery hosted on a remote server. The primary methods are implemented in the Car_Detector() class inside of cars.py. Each method (detect, compare, and analyze) uses the yolov3 deep CNN model for car detection, and requires the user to input at least two parameters: 'mode' and 'timestamp' of the video to be processed. The third optional argument for compare() and analyze() is a second timestamp for comparing parked cars or analyzing parking activity in the interval specified by timestamp : timestamp2 

Set Up
===============

- Unpack the tar archive in your preferred location:

  tar -xzvf yolov3_car_detector.tar.gz

- Build the docker image and give it the name "car-detector":
    
  cd yolov3_car_detector
  docker build -t car-detector .


Example Uses
===============

- Detection:

  python3 cars.py --mode detect --timestamp 1538076003
  False

- Compare:
    
  python3 cars.py --mode compare --timestamp 1538076179 --timestamp2 1538076183
  True
  
- Analyze:

  python3 cars.py --mode analyze --timestamp 1538076003 --timestamp2 1538078234
  Analyzing from 1538076003 to 1538078234
  Found car at 1538076175.
  Parked until 1538077874 28 minutes
  ...
  
Dependencies
===============

- reuests
- Opencv3
- libav-tools   

Notes on Development, Deployment, Testing, and Scalability
===============

- This package was developed using Python requests and OpenCV's DNN module for simplicity and quick time to implement.

- Code was developed and tested with both cv2.dnn.OPENCV_TARGET_OPENCL for GPU use locally, and OPENCV_TARGET_CPU for CPU inference in docker

- To enable Opencl in docker, you could use a container similar to: https://github.com/pkienzle/opencl_docker
  We would also need to buid opencv3/4 from source with -D WITH_OPENCL=True    

- For scalable deployment it would be ideal to implement a tensorflow or pure C++/OpenCL yolov3 to be distributed on multiple GPUS 
  An example tf implementation of Yolov3 is included in src/third_party/, barrowed from https://github.com/mystic123/tensorflow-yolo-v3
  
- This repo utilizes the full yolov3 model and weights. Performance could be improved by using the compressed (tiny-model), or by
  pruning/quantizing the weights.     
    
- Analyze() accuracey is too sensitive to object deformations (a parked car with its trunk open may cause a miss with yolov3)
  This can be due to hyperparameter choices, and can be tuned by adjusting self.nmsThreshold and self.confThreshold in cars.py
  Retraining a Yolov3 model (or other cnn) with more 'trunk-open' examples from different camera angles could improve accuracey.

- Cars ocluding the target parking space in selcted frames can also cause a yolov3 missed detection. To address this, we could
  sample frmes from the video data more densely, to try and insure we detect any frames where the car is visible.

Recommendations
===============

- To use OPENCV_TARGET_OPENCL acceleration with OpenCV dnn backend for yolov3, you should install the appropriate ocl driver for your GPU
  For example, on an AWS EC2 p2.xlarge instance with one Tesla K80 GPU, you can download the driver from:

  https://www.nvidia.com/Download/index.aspx?lang=en-us

  and install with: sh ./NVIDIA_driver.run

Python 2 or 3?
--------------

- Code developed under Python 3, tested for Python 3
  Consider Python 3 to be the default

Questions
--------------

- Please send any quations to icompute@protonmail.com
