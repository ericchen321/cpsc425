from PIL import Image, ImageDraw
import numpy as np
import random
import os.path
import pickle
import math

##############################################################################
#                        Functions for you to complete                       #
##############################################################################

def ComputeSSD(TODOPatch, TODOMask, textureIm, patchL):
	patch_rows, patch_cols, patch_bands = np.shape(TODOPatch)
	tex_rows, tex_cols, tex_bands = np.shape(textureIm)
	ssd_rows = tex_rows - 2 * patchL
	ssd_cols = tex_cols - 2 * patchL
	SSD = np.zeros((ssd_rows,ssd_cols))

	# sanity checks
	assert patch_bands == tex_bands

	# Convert TODOPatch and textureIm to float64 type
	TODOPatch_float = TODOPatch.astype(np.float64)
	textureIm_float = textureIm.astype(np.float64)

	for r in range(ssd_rows):
		for c in range(ssd_cols):
			# Compute sum square difference between textureIm and TODOPatch
			# for all pixels where TODOMask = 0, and store the result in SSD
			#
			# ADD YOUR CODE HERE
			#
			# Compute center locations of window in texture image
			rTextureImgWindowCtr = r + patchL
			cTextureImgWindowCtr = c + patchL
			# Compute SSD by summing up squared difference of each pixel
			# in the patch
			"""
			# Here we show a non-vectorized implementation
			sum_sd = 0
			for rowPatch in range(0, patch_rows):
				for colPatch in range(0, patch_cols):
					for channel in range(0, patch_bands):
						# Check if the patch is filled. If so then compute SD and add to sum; if not then do no compute
						if TODOMask[rowPatch, colPatch] == 0:
							# Compute pixel coordinates within texture image
							rowTexture = rTextureImgWindowCtr - patchL + rowPatch
							colTexture = cTextureImgWindowCtr - patchL + colPatch
							# Compute squared difference
							sd = math.pow((TODOPatch_float[rowPatch, colPatch, channel] - textureIm_float[rowTexture, colTexture, channel]), 2)
							# Sum up to SSD
							sum_sd = sum_sd + sd
			"""
			# Here is a vectorized implementation
			# Select windows from textureIm
			textureImWindow_float = textureIm_float[rTextureImgWindowCtr-patchL:rTextureImgWindowCtr+patchL+1, cTextureImgWindowCtr-patchL:cTextureImgWindowCtr+patchL+1, :]
			# Compute SD per channel, per pixel
			sd = np.square(TODOPatch_float - textureImWindow_float)
			# Set SD to 0 for pixels that have not been filled
			sd = sd[TODOMask==0]

			# Compute window SSD by summing up SD across all channels and all pixels
			SSD[r, c] = np.sum(sd)

	return SSD

def CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iMatchCenter,jMatchCenter,patchL):
	patchSize = 2 * patchL + 1
	for i in range(patchSize):
		for j in range(patchSize):
			# Copy the selected patch selectPatch into the image containing
			# the hole imHole for each pixel where TODOMask = 1.
			# The patch is centred on iPatchCenter, jPatchCenter in the image imHole
			#
			# ADD YOUR CODE HERE
			#
			# Compute pixel location in imHole
			row_imHole = iPatchCenter - patchL + i
			col_imHole = jPatchCenter - patchL + j
			# Compute pixel location in TODOMask
			row_TODOMask = i
			col_TODOMask = j
			# Compute pixel location in textureIm
			row_textureIm = iMatchCenter - patchL + i
			col_textureIm = jMatchCenter - patchL + j
			# Check if the pixel has been filled. If not then copy all channels; otherwise do not overwrite
			if TODOMask[row_TODOMask, col_TODOMask] == 1:
				imHole[row_imHole, col_imHole, :] = textureIm[row_textureIm, col_textureIm, :]
	return imHole

##############################################################################
#                            Some helper functions                           #
##############################################################################

def DrawBox(im,x1,y1,x2,y2):
	draw = ImageDraw.Draw(im)
	draw.line((x1,y1,x1,y2),fill="white",width=1)
	draw.line((x1,y1,x2,y1),fill="white",width=1)
	draw.line((x2,y2,x1,y2),fill="white",width=1)
	draw.line((x2,y2,x2,y1),fill="white",width=1)
	del draw
	return im

def Find_Edge(hole_mask):
	[cols, rows] = np.shape(hole_mask)
	edge_mask = np.zeros(np.shape(hole_mask))
	for y in range(rows):
		for x in range(cols):
			if (hole_mask[x,y] == 1):
				if (hole_mask[x-1,y] == 0 or
						hole_mask[x+1,y] == 0 or
						hole_mask[x,y-1] == 0 or
						hole_mask[x,y+1] == 0):
					edge_mask[x,y] = 1
	return edge_mask

##############################################################################
#                           Main script starts here                          #
##############################################################################

#
# Constants
#

# Change patchL to change the patch size used (patch size is 2 *patchL + 1)
patchL = 20
patchSize = 2*patchL+1

# Standard deviation for random patch selection
randomPatchSD = 14

# Display results interactively
showResults = True

#
# Read input image
#

im = Image.open('bliss.jpg').convert('RGB')
im_array = np.asarray(im, dtype=np.uint8)
imRows, imCols, imBands = np.shape(im_array)

#
# Define hole and texture regions.  This will use files fill_region.pkl and
#   texture_region.pkl, if both exist, otherwise user has to select the regions.
if os.path.isfile('fill_region.pkl') and os.path.isfile('texture_region.pkl'):
	fill_region_file = open('fill_region.pkl', 'rb')
	fillRegion = pickle.load( fill_region_file )
	fill_region_file.close()

	texture_region_file = open('texture_region.pkl', 'rb')
	textureRegion = pickle.load( texture_region_file )
	texture_region_file.close()
else:
	# ask the user to define the regions
	print("Specify the fill and texture regions using polyselect.py")
	exit()

#
# Get coordinates for hole and texture regions
#

fill_indices = fillRegion.nonzero()
nFill = len(fill_indices[0])                # number of pixels to be filled
iFillMax = max(fill_indices[0])
iFillMin = min(fill_indices[0])
jFillMax = max(fill_indices[1])
jFillMin = min(fill_indices[1])
assert((iFillMin >= patchL) and
		(iFillMax < imRows - patchL) and
		(jFillMin >= patchL) and
		(jFillMax < imCols - patchL)) , "Hole is too close to edge of image for this patch size"

texture_indices = textureRegion.nonzero()
iTextureMax = max(texture_indices[0])
iTextureMin = min(texture_indices[0])
jTextureMax = max(texture_indices[1])
jTextureMin = min(texture_indices[1])
textureIm   = im_array[iTextureMin:iTextureMax+1, jTextureMin:jTextureMax+1, :]
texImRows, texImCols, texImBands = np.shape(textureIm)
assert((texImRows > patchSize) and
		(texImCols > patchSize)) , "Texture image is smaller than patch size"

#
# Initialize imHole for texture synthesis (i.e., set fill pixels to 0)
#

imHole = im_array.copy()
imHole[fill_indices] = 0

#
# Is the user happy with fillRegion and textureIm?
#
if showResults == True:
	# original
	im.show()
	# convert to a PIL image, show fillRegion and draw a box around textureIm
	im1 = Image.fromarray(imHole).convert('RGB')
	im1 = DrawBox(im1,jTextureMin,iTextureMin,jTextureMax,iTextureMax)
	im1.show()
	print("Are you happy with this choice of fillRegion and textureIm?")
	Yes_or_No = False
	while not Yes_or_No:
		answer = input("Yes or No: ")
		if answer == "Yes" or answer == "No":
			Yes_or_No = True
	assert answer == "Yes", "You must be happy. Please try again."

#
# Perform the hole filling
#

while (nFill > 0):
	print("Number of pixels remaining = " , nFill)

	# Set TODORegion to pixels on the boundary of the current fillRegion
	# NOTE: fillRegion is an array of binary values - 1 means the pixel is part of a hole, 0 means not
	TODORegion = Find_Edge(fillRegion) # NOTE: TODORegion is an array of pixels with same size as fillRegion st. each boundary pixel is marked 1, other pixels marked as 0
	edge_pixels = TODORegion.nonzero() # NOTE: a tuple of positions of edge pixels
	nTODO = len(edge_pixels[0])

	# NOTE: in this loop, we synthesize texture for one contour
	while(nTODO > 0):

		# Pick a random pixel from the TODORegion
		index = np.random.randint(0,nTODO)
		iPatchCenter = edge_pixels[0][index]
		jPatchCenter = edge_pixels[1][index]

		# Define the coordinates for the TODOPatch
		TODOPatch = imHole[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1,:] # NOTE: TODOPatch is a window of the image to be filled, centered around one edge pixel
		TODOMask = fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] # NOTE: TODOMask is a window of fillRegion, centered around the same edge pixel

		#
		# Compute masked SSD of TODOPatch and textureIm
		#
		# NOTE: ssdIm is an array of SSDs of each comparison between TODOPatch and one window from the texture image
		ssdIm = ComputeSSD(TODOPatch, TODOMask, textureIm, patchL)

		# Randomized selection of one of the best texture patches
		ssdIm1 = np.sort(np.copy(ssdIm),axis=None)
		ssdValue = ssdIm1[min(round(abs(random.gauss(0,randomPatchSD))),np.size(ssdIm1)-1)]
		ssdIndex = np.nonzero(ssdIm==ssdValue) # NOTE: ssdIndex is an array of central pixel locations (on texture image, un-offseted by patchL) of neighborhoolds where SSD==the value selected. It's a list since there may be multiple neighborhoods yielding the same SSD
		iSelectCenter = ssdIndex[0][0]
		jSelectCenter = ssdIndex[1][0]

		# adjust i, j coordinates relative to textureIm
		iSelectCenter = iSelectCenter + patchL # NOTE: i pos of center of finally selected window on texture image
		jSelectCenter = jSelectCenter + patchL # NOTE: j pos of center of finally selected window on texture image
		selectPatch = textureIm[iSelectCenter-patchL:iSelectCenter+patchL+1,jSelectCenter-patchL:jSelectCenter+patchL+1,:]

		#
		# Copy patch into hole
		#
		# NOTE: copying an entire window from the texture image to the window centered around the picked pixel
		imHole = CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iSelectCenter,jSelectCenter,patchL)

		# Update TODORegion and fillRegion by removing locations that overlapped the patch
		TODORegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0
		fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0

		edge_pixels = TODORegion.nonzero()
		nTODO = len(edge_pixels[0])
		print("Number of pixels todo = " , nTODO)

	fill_indices = fillRegion.nonzero()
	nFill = len(fill_indices[0])

#
# Output results
#
if showResults == True:
	Image.fromarray(imHole).convert('RGB').show()
Image.fromarray(imHole).convert('RGB').save('results.jpg')
