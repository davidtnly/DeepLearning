import matplotlib
import PIL
import scipy

""" Decoding Images
Images may have various extensions — JPG, PNG, TIFF are common. This post focuses on JPG images only, but the process 
for other image formats should not be very different. The first step in the process is to read the image.

An image with a JPG extension is stored in memory as a list of dots, known as pixels. A pixel, or a picture element, 
represents a single dot in an image. The color of the dot is determined by a combination of three values — its three 
component colors (Red, Blue and Green). The color of the pixel is essentially a combination of these three component 
colors.
"""

