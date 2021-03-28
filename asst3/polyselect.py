from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pickle

'''
The MIT License (MIT)

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

polyselect.py allows you to choose a polygon from an image region using
nothing other than your mouse.

Written by Julieta, Jan 2015.
'''

class LineBuilder:

	def __init__(self, line):
		# Prepare for the first click.
		self.line = line
		self.first_click = True
		# Add click listener
		self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
		# Add close listener
		self.xs = list()
		self.ys = list()

	def __call__(self, event):

		# Initialize with the first click.
		if self.first_click:
			self.first_click = False
			self.xs.append( int(event.xdata) )
			self.ys.append( int(event.ydata) )
			self.line.set_data(self.xs, self.ys)
			return

		# Handle further clicks.
		if event.inaxes!=self.line.axes: return
		self.xs.append( int(event.xdata) )
		self.ys.append( int(event.ydata) )
		self.line.set_data(self.xs, self.ys)
		self.line.figure.canvas.draw()


# Function that handles closing the viewer
def handle_close(event):
	xs = event.canvas.figure.linebuilder.xs
	ys = event.canvas.figure.linebuilder.ys
	fillPolyPoint = list()
	for i in range( len(xs) ):
		fillPolyPoint.append( xs[i] )
		fillPolyPoint.append( ys[i] )

	assert len( fillPolyPoint ) >= 6, "A polygon requires at least 3 points"
	img = Image.new('L', (ncols, nrows), 0)
	ImageDraw.Draw(img).polygon(fillPolyPoint, outline=1, fill=1)
	fillRegion = np.array(img, dtype=np.uint8)

	# Save the pickle
	ff = open( fname, 'wb')
	pickle.dump( fillRegion, ff, -1 )
	ff.close()

	print('Saved region to {0}!'.format( fname ))


# === Script execution starts here ===
# imname is the name of the image file that you want to read.
imname = 'bliss.jpg'

# === Read the image
im = Image.open( imname ).convert('RGB')
im_array = np.asarray( im, dtype=np.uint8 )
nrows, ncols, _ = im_array.shape

print('Would you like to select the region to be filled (0) or the sample texture region (1)?')

Zero_or_One = False
while not Zero_or_One:
    answer = input("0 or 1: ")
    if answer == "0" or answer == "1":
            Zero_or_One = True

if answer == "0":
        fname = 'fill_region.pkl'
else:
        fname  = 'texture_region.pkl'
        print('Note: Code in Holefill.py forces the texture region to be rectangular')

print('Please use your mouse to specify the region that you want for {0}'.format( fname ))
print('(Click to select each polygon vertex. Close the window to complete and save the polygon.)')

# === Create display
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([ 0, ncols])
ax.set_ylim([ 0, nrows])
ax.invert_yaxis()

# === Display the image
ax.imshow( im_array )
ax.set_title('click to build line segments')

# === Add listener for close event
fig.canvas.mpl_connect('close_event', handle_close)

line, = ax.plot([0], [0])  # empty line
fig.linebuilder = LineBuilder(line)

plt.show()

