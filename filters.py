import cv2, argparse
import numpy as np

def makeCartoon(original):

  # Make a copy of the origianl image to work with
  img = np.copy(original)

  # Convert image to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply gaussian filter to the grayscale image
  imgGray = cv2.GaussianBlur(imgGray, (3,3), 0)

  # Detect edges in the image and threshold it
  edges = cv2.Laplacian(imgGray, cv2.CV_8U, ksize=5)
  edges = 255 - edges
  ret, edgeMask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
  
  # Apply Edge preserving filter to get the heavily blurred image
  imgBilateral = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)

  # Create a outputmatrix
  output = np.zeros(imgGray.shape)
  
  # Combine the cartoon and edges 
  output = cv2.bitwise_and(imgBilateral, imgBilateral, mask=edgeMask)

  return output



def clarendon(original):

  img = np.copy(original)

  # Separate the channels
  bChannel = img[:,:,0]
  gChannel = img[:,:,1]
  rChannel = img[:,:,2]

  # Specifying the x-axis for mapping
  xValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])

  # Specifying the y-axis for different channels
  rCurve = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249 ])
  gCurve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])
  bCurve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])

  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)
  bLUT = np.interp(fullRange, xValues, bCurve )
  gLUT = np.interp(fullRange, xValues, gCurve )
  rLUT = np.interp(fullRange, xValues, rCurve )

  # Applying the mapping to the image using LUT
  bChannel = cv2.LUT(bChannel, bLUT)
  gChannel = cv2.LUT(gChannel, gLUT)
  rChannel = cv2.LUT(rChannel, rLUT)

  # Converting back to uint8
  img[:,:,0] = np.uint8(bChannel)
  img[:,:,1] = np.uint8(gChannel)
  img[:,:,2] = np.uint8(rChannel)

  return img


def adjustSaturation(original, saturationScale = 1.0):
  img = np.copy(original)

  # Convert to HSV color space
  hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

  # Convert to float32
  hsvImage = np.float32(hsvImage)

  # Split the channels
  H, S, V = cv2.split(hsvImage)

  # Multiply S channel by scaling factor 
  S = np.clip(S * saturationScale , 0, 255)

  # Merge the channels and show the output
  hsvImage = np.uint8( cv2.merge([H, S, V]) )

  imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
  return imSat

def moon(original):

  img = np.copy(original)

  # Specifying the x-axis for mapping
  origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255 ])
  
  # Specifying the y-axis for mapping
  Curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255  ])

  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)

  LUT = np.interp(fullRange, origin, Curve )

  # Applying the mapping to the L channel of the LAB color space
  labImage = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
  labImage[:,:,0] = cv2.LUT(labImage[:,:,0], LUT)
  img = cv2.cvtColor(labImage,cv2.COLOR_LAB2BGR)

  # Desaturating the image
  img = adjustSaturation(img,0.01)

  return img


def adjustContrast(original, scaleFactor): 
  img = np.copy(original)

  # Convert to YCrCb color space
  ycbImage = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

  # Convert to float32 since we will be doing multiplication operation
  ycbImage = np.float32(ycbImage)

  # Split the channels
  Ychannel, Cr, Cb = cv2.split(ycbImage)

  # Scale the Ychannel 
  Ychannel = np.clip(Ychannel * scaleFactor , 0, 255)

  # Merge the channels and show the output
  ycbImage = np.uint8( cv2.merge([Ychannel, Cr, Cb]) )

  img = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

  return img


def applyVignette(original, vignetteScale):
  img = np.copy(original)

  # convert to float
  img = np.float32(img)
  rows,cols = img.shape[:2]

  # Compute the kernel size from the image dimensions
  k = np.min(img.shape[:2])/vignetteScale

  # Create a kernel to get the halo effect 
  kernelX = cv2.getGaussianKernel(cols,k)
  kernelY = cv2.getGaussianKernel(rows,k)

  # generating vignette mask using Gaussian kernels
  kernel = kernelY * kernelX.T

  # Normalize the kernel
  mask = 255 * kernel / np.linalg.norm(kernel)

  mask = cv2.GaussianBlur(mask, (51,51), 0)

  # Apply the halo to all the channels of the image
  img[:,:,0] += img[:,:,0]*mask
  img[:,:,1] += img[:,:,1]*mask
  img[:,:,2] += img[:,:,2]*mask

  img = np.clip(img/2, 0, 255)

  # cv2.imshow("mask",mask)
  # cv2.waitKey(0)
  # cv2.imwrite("results/vignetteMask.jpg", 255*mask)

  return np.uint8(img)

def xpro2(original, vignetteScale=3):

  img = np.copy(original)

  # Applying a vignette with some radius
  img = applyVignette(img, vignetteScale) 

  # Separate the channels
  bChannel = img[:,:,0]
  gChannel = img[:,:,1]
  rChannel = img[:,:,2]

  # Specifying the x-axis for mapping
  originalR = np.array([0, 42, 105, 148, 185, 255])
  originalG = np.array([0, 40, 85, 125, 165, 212, 255])
  originalB = np.array([0, 40, 82, 125, 170, 225, 255 ])
  
  # Specifying the y-axis for mapping
  rCurve = np.array([0, 28, 100, 165, 215, 255 ])
  gCurve = np.array([0, 25, 75, 135, 185, 230, 255 ])
  bCurve = np.array([0, 38, 90, 125, 160, 210, 222])
  
  # Creating the LUT to store the interpolated mapping
  fullRange = np.arange(0,256)
  bLUT = np.interp(fullRange, originalB, bCurve )
  gLUT = np.interp(fullRange, originalG, gCurve )
  rLUT = np.interp(fullRange, originalR, rCurve )

  # Applying the mapping to the image using LUT
  bChannel = cv2.LUT(bChannel, bLUT)
  gChannel = cv2.LUT(gChannel, gLUT)
  rChannel = cv2.LUT(rChannel, rLUT)

  # Converting back to uint8
  img[:,:,0] = np.uint8(bChannel)
  img[:,:,1] = np.uint8(gChannel)
  img[:,:,2] = np.uint8(rChannel) 

  # Adjusting the contrast a bit - just for fun!
  img = adjustContrast(img,1.2)

  return img



def colorDodge(top, bottom):
  
  # divid the bottom by inverted top image and scale back to 250
  output = cv2.divide(bottom, 255 - top , scale = 256)
  
  return output


def sketchPencilUsingBlending(original,kernelSize = 21):
  img = np.copy(original)

  # Convert to grayscale
  imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Invert the grayscale image
  imgGrayInv = 255 - imgGray
  
  # Apply GaussianBlur
  imgGrayInvBlur =  cv2.GaussianBlur(imgGrayInv, (kernelSize,kernelSize), 0)

  # blend using color dodge
  output = colorDodge(imgGrayInvBlur, imgGray)

  return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

