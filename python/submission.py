# Requires the following to run
# DATA_PATH + "videos/soccer-ball.mp4"
# MODEL_PATH + "coco.names"
# MODEL_PATH + "yolov3-tiny.cfg"                 # Using tiny-yolo!
# MODEL_PATH + "yolov3-tiny.weights"

import sys
import os.path
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
DATA_PATH="../data/"
MODEL_PATH="../models/"
#from dataPath import DATA_PATH
#from dataPath import MODEL_PATH

#%matplotlib inline
#import matplotlib
#matplotlib.rcParams['figure.figsize'] = (25.0,25.0)
#matplotlib.rcParams['image.cmap'] = 'gray'

# Functions

# Initialize tracker
def initTracker(tracker_type):

    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
            
    return tracker
            
# Grow bounding box due to fast-yolo small bounding boxes
def growBoundingBox(bbox, scaleFactor):
    cx = bbox[0] + bbox[2] // 2
    cy = bbox[1] + bbox[3] // 2
    
    # This is the weakest part of this implementation; prevents it from working on other videos
    MINSIZE = 130
    MAXSIZE = 300
    
    maxDim = max(bbox[2], bbox[3])
    newSize = maxDim * scaleFactor;
    
    if (newSize == 0):
        pass    
    elif (newSize > MAXSIZE):
        newSize = MAXSIZE
    elif (newSize < MINSIZE):
        newSize = MINSIZE
    
    return (cx - newSize // 2, cy - newSize // 2, newSize, newSize)
    
def initTrackerWithGrowingBox(tracker_type, frame, detectBox):
    scaleFactor = 1.0
    tracker = initTracker(tracker_type)
    trackerOk = tracker.init(frame, growBoundingBox(detectBox, scaleFactor))
    trackerOk, trackBox = tracker.update(frame)
    while (not trackerOk and scaleFactor <= 3.0):
        scaleFactor += 0.2
        tracker = initTracker(tracker_type)
        trackerOk = tracker.init(frame, growBoundingBox(detectBox, scaleFactor))
        trackerOk, trackBox = tracker.update(frame)
        
    scaleFactor += 1.0
    tracker = initTracker(tracker_type)
    trackerOk = tracker.init(frame, growBoundingBox(detectBox, scaleFactor))
    trackerOk, trackBox = tracker.update(frame)
    
    return tracker, trackerOk, trackBox
    
def drawBox(frame, bbox, color):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    
def distanceBetweenBoxes(box1, box2):
    x1 = box1[0] + box1[2] / 2.0
    y1 = box1[1] + box1[3] / 2.0
    x2 = box2[0] + box2[2] / 2.0
    y2 = box2[1] + box2[3] / 2.0

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                if classId != 32:# and classId != 0: # 32 is sports-ball, 0 is person
                    continue
                
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    
    if len(boxes) == 0:
        return (0, 0, 0, 0), 0
    
    # Find the largest bounding box
    maxWidth = 0
    maxIndex = 0
    i = 0
    for box in boxes:
        if (box[2] > maxWidth or box[3] > maxWidth):
            maxWidth = max(box[2], box[3])
            maxIndex = i
        i += 1

    # Make it square
    boxes[maxIndex][2] = maxWidth
    boxes[maxIndex][3] = maxWidth

    return boxes[maxIndex], confidences[maxIndex]

    # # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # # lower confidences.
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # for i in indices:
        # i = i[0]
        # box = boxes[i]
        # left = box[0]
        # top = box[1]
        # width = box[2]
        # height = box[3]
        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        
        # return boxes, confidences
    
    # return (0, 0, 0, 0), 0.0
    
# Deep Neural Network setup

# Load names of classes
classesFile = MODEL_PATH + "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
#modelConfiguration = MODEL_PATH + "yolov3.cfg"
#modelWeights = MODEL_PATH + "yolov3.weights"
modelConfiguration = MODEL_PATH + "yolov3-tiny.cfg"
modelWeights = MODEL_PATH + "yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.9        # Non-maximum suppression threshold
#inpWidth = 416            # Width of network's input image
#inpHeight = 416           # Height of network's input image
inpWidth = 320            # Width of network's input image
inpHeight = 320           # Height of network's input image

trackIndex = 4;
videoIndex = 0;
tracker_types = ['MIL', 'KCF', 'MEDIANFLOW', 'CSRT', 'MOSSE']
videos = ['soccer-ball', 'hockey', 'cycle', 'meeting']

# Read and write video
videoname = videos[videoIndex];
video = cv2.VideoCapture(DATA_PATH + "videos/" + videoname + ".mp4")
fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
outSize = [round(x) for x in [video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)]]
outVideo = cv2.VideoWriter(DATA_PATH + "videos/soccer_track.mp4", fourcc, 25.0, tuple(outSize), True)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

# Windows - Attempt to make the video window appear in the forefront; works half the time
windowName = videoname;
view_window = cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)  

# Set up tracker.
tracker_type = tracker_types[trackIndex]

# Define a few colors for drawing
red = (0,0,255)
blue = (255,128,0)
green = (0,255,0)

count = 0
trackerOk = False
while True:
    # Read a new frame
    ok, frame = video.read()   
    if not ok:
        break

    detectConf = 0
    if (not trackerOk or count % 10 == 0):
        
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        detectBox, detectConf = postprocess(frame, outs)
        detectTiming, _ = net.getPerfProfile()
        
        # Draw rectangle
        p1 = (int(detectBox[0]), int(detectBox[1]))
        p2 = (int(detectBox[0] + detectBox[2]), int(detectBox[1] + detectBox[3]))
        cv2.rectangle(frame, p1, p2, blue, 2, 1)    
        # if trackerOk:
            # # Tracking success
            # pass
        # else:
            # # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)        

    # Start timer
    timer = cv2.getTickCount()
    
    # Update tracker
    if (trackerOk and tracker != None):
        trackerOk, trackBox = tracker.update(frame)
    
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    trackTime = 1000 / fps;
    
    # Initialize tracker
    if (not trackerOk and detectConf > 0):
        # Reacquire using detector
        #tracker = initTracker(tracker_type)
        #trackerOk = tracker.init(frame, tuple(detectBox))
        #trackerOk = tracker.init(frame, growBoundingBox(detectBox, scaleFactor))
        tracker, trackerOk, trackBox = initTrackerWithGrowingBox(tracker_type, frame, detectBox)
        #trackerOk, trackBox = tracker.update(frame)
        if (trackerOk):
            #print("Frame: " + str(count) + "  Initialized tracker")
            drawBox(frame, trackBox, green)
        else:
            print("Frame: " + str(count) + "  ERROR - Unable to initialize tracker with given bounding box")
            cv2.putText(frame, "Tracking failure detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)
    elif (not trackerOk and detectConf <= 0):
        print("Frame: " + str(count) + "  ERROR - Tracking and detection failed")
        cv2.putText(frame, "Tracking failure detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)
    elif (trackerOk and detectConf > 0):

        d = distanceBetweenBoxes(detectBox, trackBox)
        if (d > 50):
            
            newTracker, newTrackerOk, newTrackBox = initTrackerWithGrowingBox(tracker_type, frame, detectBox)
            if (newTrackerOk):
                print("Frame: " + str(count) + "  Reacquire tracker due to distance " + str(d))
                tracker = newTracker
                trackerOk = newTrackerOk
                trackBox = newTrackBox
                #trackerOk, trackBox = tracker.update(frame)
        
        drawBox(frame, trackBox, green)
    elif (trackerOk and detectConf <= 0):
        drawBox(frame, trackBox, green)  
    
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2);
    
    # Display FPS on frame
    label2 = "Tracker time: %.2f ms" % trackTime
    cv2.putText(frame, label2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2);

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    label = 'Detect time: %.2f ms' % (detectTiming * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
  
    # Write frame
    outVideo.write(frame);
  
    # Display frame
    cv2.imshow(windowName, frame);
    if (trackerOk):
        cv2.waitKey(1);
    else:
        cv2.waitKey(1); # slow-mo
        
    
    # Exit
    count += 1
    if count == 750:
        break

cv2.waitKey(3000)
video.release()
outVideo.release()
cv2.destroyAllWindows()
