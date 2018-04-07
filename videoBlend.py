import cv2, argparse
import numpy as np
from filters import sketchPencilUsingBlending, xpro2, moon, clarendon, makeCartoon

filter_type = 0
max_value = 255
max_type = 4


# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


windowName = "Video Demo"
filterType = "Type: \n 0: PencilSketch \n 1: Cartoon \n 2: Clarendon \n 3: Moon Filter \n 4: XPro II"
# Call the function to initialize
# 0: PencilSketch
# 1: Cartoon
# 2: Clarendon
# 3: Moon Filter
# 4: XPro II

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

# Define filter type 
def filterTypeDemo(*args):
    global filter_type
    filter_type = args[0]

    

while(True):
  ret, frame = cap.read()

  if ret == True: 
    # Create Trackbar to choose type of Image Filter
    cv2.createTrackbar(filterType, windowName, filter_type, max_type, filterTypeDemo)

    # Write the frame into the file 'output.avi'

    if filter_type == 0:
      output = sketchPencilUsingBlending(frame) 
    elif filter_type == 1:
      output = makeCartoon(frame)
    elif filter_type == 2:
      output = clarendon(frame)
    elif filter_type == 3:
      output = moon(frame)
    elif filter_type == 4:
      output = xpro2(frame)

    out.write(output)

    # Display the resulting frame    
    cv2.imshow(windowName,output)

    # Press ESC on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == 27:
      break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 

