# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:37:45 2018

@author: aakash

Link for reference ASL signs
http://www.lifeprint.com/asl101/pages-signs/n/numbers1-10.htm
C:\OpenCV\Project Folder\signlanguageabc02.jpg

"""


import cv2
import sys
import os
import numpy as np
from datetime import datetime


if __name__ == '__main__':
  
  print("Number of arguments: ", len(sys.argv))
  
  if len(sys.argv) > 1:
      dataPoint = sys.argv[1].upper()
      print("arguments[1]: ", dataPoint)
      
      if not os.path.exists("signs"):
          os.makedirs("signs")
      dataDir = os.path.join("signs", dataPoint)    
      if not os.path.exists(dataDir):
          os.makedirs(dataDir)
  else:
      dataPoint = ' '
      print("Next time enter a number [0-9] or an alphabet [a-z]")
  
      
  
  # Start webcam
  cap = cv2.VideoCapture(0)
  
  # We convert the resolutions from float to integer.
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  
  print(frame_height)
  print(frame_width)
  

  # Check if webcam opens
  if (cap.isOpened()== False):
    print("Error opening video stream or file")
  
  # Window for displaying output
  cv2.namedWindow("Skin Detection")
  
  # Recording flag
  recordFlag = False
  count = 0
  
  while(1):

    # Read frame
    ret, image = cap.read()

    # Split frame into r, g and b channels
    b,g,r = cv2.split(image)

    # Write the frame into the file 'output.avi'
    cv2.rectangle(image, ( 20 , 20 ), ( int(frame_width*0.45) , int(frame_height*0.45) ), ( 255 , 255 , 255 ), thickness=+ 2 , lineType=cv2.LINE_8)
    
    if( recordFlag == False):
        stringDisplay = "Press Space to START recording OR Esc to quit. Alphabet: " + dataPoint + " Count: " + str(count)
        cv2.putText(image, stringDisplay , (20 , int(frame_height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , ( 0 , 255 , 0 ), 2 )
    else:
        stringDisplay = "Press Space to STOP recording OR Esc to quit. Alphabet: " + dataPoint + " Count: " + str(count)
        cv2.putText(image, stringDisplay , (20 , int(frame_height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , ( 0 , 255 , 0 ), 2 )
    
    # Display results
    cv2.imshow("Skin Detection",image)
    
    keyPress = cv2.waitKey(10) & 0xFF
    
    if keyPress == 27:
      break
    
    if( keyPress == 32):
        if recordFlag == True:
            recordFlag = False
        else:
            recordFlag = True
    
    if( keyPress in range(65,91) or keyPress in range(97, 123) or keyPress in range(48, 58)):
        dataPoint = chr(keyPress).upper()
        dataDir = os.path.join("signs", dataPoint)
        if not os.path.exists(dataDir):
          os.makedirs(dataDir)
          count = 0
        else:
          count = len(os.listdir(dataDir))
    
    
    if recordFlag == True:
        
        outImage = image[20:int(frame_height*0.45) , 20:int(frame_width*0.45) ]
        
        timeStamp = str(datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")
        fileName = dataPoint + timeStamp +".jpg"
        fileName = os.path.join(dataDir, fileName)
        
        print(fileName)
        cv2.imwrite( fileName, outImage )
        count +=1

  cap.release()
  cv2.destroyAllWindows()
