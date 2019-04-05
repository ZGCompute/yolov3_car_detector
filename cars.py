import os
import argparse
from src import cache, compare
import cv2
import subprocess
import numpy as np
from math import ceil
from collections import deque

HOME = os.getcwd()
TENSORFLOW_DIR = os.path.join(HOME, "src/third_party/tensorflow-yolo-v3/")

class Car_Detector:

    def __init__(self, args):
        self.args = args
        self.timestamp = args.timestamp
        if(args.mode == 'compare' or args.mode == 'analyze'):
            self.t2 = args.timestamp2
        self.cache = cache
        self.confThreshold = 0.45  
        self.nmsThreshold = 0.4   
        self.histThreshold = 0.33
        self.inpWidth = 416        
        self.inpHeight = 416
        self.classesFile = os.path.join(TENSORFLOW_DIR, "coco.names")
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.modelConfiguration = os.path.join(TENSORFLOW_DIR, "yolov3.cfg")
        self.modelWeights = os.path.join(TENSORFLOW_DIR, "yolov3.weights")
        self.setup_model()

        self.img = None # OpenCV image reference
        self.last_detected_img = None

    def download_model_files(self):
        '''
        Download yolov3 cfg and weights
        '''
        if not os.path.isfile(self.modelWeights):
            subprocess.call('wget -q https://pjreddie.com/media/files/yolov3.weights', shell=True)
            subprocess.call('wget -q https://raw.githubusercontent.com/marvis/pytorch-yolo3/master/cfg/yolov3.cfg', shell=True)
            subprocess.call('mv yolov3.weights src/third_party/tensorflow-yolo-v3/yolov3.weights', shell=True)
            subprocess.call('mv yolov3.cfg src/third_party/tensorflow-yolo-v3/yolov3.cfg', shell=True)
        
    def setup_model(self):
        '''
        Setup detection CNN (Yolov3) using OpenCv DNN
        '''
        self.download_model_files()
        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def diff_unixts(self, timestamp1, timestamp2):
        ''' 
        Get the time tifference in minutes between two unix timestamps
        '''
        return ceil(abs((timestamp2 - timestamp1) / 60))

    def drawPred(self, classId, conf, left, top, right, bottom):
        '''
         Draw a bounding box. 
        '''
        cv2.rectangle(self.img, (left, top), (right, bottom), (0, 0, 255))
        label = '%.2f' % conf
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(self.img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    def getOutputsNames(self):
        '''
        Get the names of all the layers in the network
        '''
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outp
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def postprocess(self, outs):
        '''
        perform non-max supression on yolov3 detectiion outputs
        '''
        frameHeight = self.img.shape[0]
        frameWidth = self.img.shape[1]
 
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and classId == 2: #for cars only
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    if ((center_x < 260 and center_x > 180) and (center_y < 260 and center_y > 180)): #for our parking ROI
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
 
        # Perform non maximum suppression to eliminate redundant overlapping boxes 
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        return boxes
            
    def detect(self, timestamp):
        '''
        Perform inference on input image using yolov3
        '''
        timestamp = str(timestamp)
        cache_dir = self.cache.get_cache_dir()
        filename = os.path.join(cache_dir, (timestamp + '.jpg'))
        
        if not os.path.isfile(filename):
            self.cache.get_image(timestamp)
        if not os.path.isfile(filename): #get_image failed
            return False
        
        self.img = cv2.imread(filename)
        blob = cv2.dnn.blobFromImage(self.img, 1/255, (416, 416), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames())
        
        # Remove the bounding boxes with low confidence
        boxes = self.postprocess(outs)
        if (len(boxes) > 0):
            if (self.args.mode == 'detect'):
                print("True")
                outfile = timestamp + '_detect.jpg' 
                cv2.imwrite(outfile, self.img.astype(np.uint8));
                                                    
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(self.img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            return True

        else:
            if (self.args.mode == 'detect'):
                print("False")        
            return False

    def analyze(self):
        '''
        Run analysis(detections) on a range of input timestamps from start:end       
        '''
        print("Analyzing from %s to %s" %(self.timestamp, self.t2))

        # enqueue all dates to be processed
        q = deque([date for date in range(int(self.timestamp), int(self.t2), 8)])
        found = []   
        parked = None     

        # Run detection on frames sampled from video every 4 seconds in input range 
        while (q):
            date = q.popleft()
            res = self.detect(date)
            if (res == True):                            
                if (len(found) == 0 or parked == None):  
                    parked = date                        
                    print("Found car at %s" %(date))
                    found.append(date)
                    self.last_detected_img = self.img

                elif (parked != None):                         
                    same = self.compare(found[-1], date)
                    if (same != True):
                        diff = self.diff_unixts(date, parked)
                        print("Parked until %s %d minutes" %(date, diff))
                        outfile = str(found[-1]) + '_%dminutes.jpg' %(diff)
                        cv2.imwrite(outfile, self.last_detected_img.astype(np.uint8));
                        
                        print("...wrote %s" %(outfile))
                        parked = date
                        found.append(date)
                        print("Found car at %s" %(date))
                        
            elif (res == False and parked != None):
                diff = self.diff_unixts(date, parked)
                if (diff <=4):
                    continue
                
                print("Parked until %s %d minutes" %(date, diff))
                outfile = str(found[-1]) + '_%dminutes.jpg' %(diff)
                cv2.imwrite(outfile, self.last_detected_img.astype(np.uint8));
                print("...wrote %s" %(outfile))
                                
                parked = None

                        
    def compare(self, t1, t2):
        '''
        Run comprison metric (Bhattacharyya Hist method) on two input timestamps 
        to determine if detected car is the same       
        '''
        cache_dir = self.cache.get_cache_dir()
        f1 = os.path.join(cache_dir, (str(t1) + '.jpg'))
        f2 = os.path.join(cache_dir, (str(t2) + '.jpg'))
        if not os.path.isfile(f1):
            self.cache.get_image(self.timestamp)
        if not os.path.isfile(f2):
            self.cache.get_image(self.t2)

        cmp = compare.CompareImage(f1, f2)
        score = cmp.compare_image()
        
        if (score > self.histThreshold):
            if (self.args.mode == 'compare'):            
                print("False")
            return False
        else:
            if (self.args.mode == 'compare'):
                print("True")
            return True
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',\
    help='mode to run inference in - detect compare or analyze')
    parser.add_argument('-t', '--timestamp',\
    help='unix timestamp of .ts file to process')
    parser.add_argument('-t2','--timestamp2', required=False,\
    help='unix timestamp2 of .ts file range to compare/analyze')
    
    args = parser.parse_args()
    model = Car_Detector(args)
    if (args.mode == 'detect'):
        model.detect(args.timestamp)
    if (args.mode == 'compare'):
        model.compare(args.timestamp, args.timestamp2)
    if (args.mode == 'analyze'):
        model.analyze()
