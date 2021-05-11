# Requires the following to run
# DATA_PATH + "videos/soccer-ball.mp4"
# MODEL_PATH + "coco.names"
# MODEL_PATH + "yolov3-tiny.cfg"                 # Using tiny-yolo!
# MODEL_PATH + "yolov3-tiny.weights"

import sys                          # Python 3.8.5
import os.path
import numpy as np                  # 1.19.2
import cv2                          # 4.5.2.52
import matplotlib.pyplot as plt     # 3.3.4

DATA_PATH="../data/"
MODEL_PATH="../models/"

# Initialize tracker
def initTracker(tracker_type):
    global tracker
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
            
# Grow bounding box due to fast-yolo small bounding boxes
def growBoundingBox(bbox):
    # Center x,y from upper-left corner and width,height
    cx = bbox[0] + bbox[2] // 2
    cy = bbox[1] + bbox[3] // 2
    
    # This is the weakest part of this implementation; prevents it from working on other videos
    MINSIZE = 130
    MAXSIZE = 300
    
    maxDim = max(bbox[2], bbox[3])
    newSize = maxDim * 2.2;
    
    if (newSize == 0):
        pass    
    elif (newSize > MAXSIZE):
        newSize = MAXSIZE
    elif (newSize < MINSIZE):
        newSize = MINSIZE
    
    return (cx - newSize // 2, cy - newSize // 2, newSize, newSize)  

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
    foundball = False
    for out in outs:
        if foundball == True:
            break
        
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
                    foundball = True
                    break

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        
        return (left, top, width, height), confidences[i]
    
    return (0, 0, 0, 0), 0.0
    
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
nmsThreshold = 0.2        # Non-maximum suppression threshold
#inpWidth = 416            # Width of network's input image
#inpHeight = 416           # Height of network's input image
inpWidth = 320            # Width of network's input image
inpHeight = 320           # Height of network's input image

trackIndex = 1;
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

    if (not trackerOk):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        bbox, confidence_yolo = postprocess(frame, outs)
        bbox = growBoundingBox(bbox)
        bbox_yolo = bbox
        t, _ = net.getPerfProfile()

    # Start timer
    timer = cv2.getTickCount()    
    
    # Initialize tracker
    if (not trackerOk and confidence_yolo > 0):
        initTracker(tracker_type)
        tracker.init(frame, [int(i) for i in bbox_yolo])
        trackerOk = True
        print("Frame: " + str(count) + "  -----------------------> Object detection <----------------------")
    elif (not trackerOk):
        print("Frame: " + str(count) + "  ERROR - Unable to track")
    else:
        print("Frame: " + str(count) + "  Object tracking")
        trackerOk, bbox = tracker.update(frame)
    
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    trackTime = 1000 / fps;

    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, green, 2, 1)    
    if trackerOk:
        # Tracking success
        pass
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2);
    
    # Display FPS on frame
    label2 = "Tracker time: %.2f ms" % trackTime
    cv2.putText(frame, label2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2);

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    label = 'Detect time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
    
    # Write frame
    outVideo.write(frame);
  
    # Display frame
    cv2.imshow(windowName, frame);
    cv2.waitKey(1);
    
    # Exit
    count += 1
    if count == 75000000:
        break

cv2.waitKey(1000)
video.release()
outVideo.release()
cv2.destroyAllWindows()
