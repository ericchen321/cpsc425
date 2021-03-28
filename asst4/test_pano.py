import cv2
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import hw_utils as utils

def main():
    path = './data/'
    image_list = ['Hanging1', 'Hanging2']
    image_list = [op.join(path, im) for im in image_list]
    # the dimension of the canvas (numpy array)
    # to which we are copying images.
    canvas_width = 1000
    canvas_height = 600

    # some precomputed data for sanity check
    with open('./data/test.pkl', 'rb') as f:
        test_dict = pkl.load(f)
    h_gt = test_dict['h']  # the homograph matrix we computed

    # matches between the source and the refence image
    xy_src = test_dict['xy_src']  # (match, 2)
    xy_ref = test_dict['xy_ref']  # (match, 2)

    # image_list should store both the reference and the source images
    ref_image = image_list[0]  # first element is the reference image
    source_image = image_list[1]

    # compute the homography matrix to transform the source to the reference
    h, _ = cv2.findHomography(xy_src, xy_ref)

    # The current computed value should equal to our precomputed one
    norm_diff = ((h-h_gt)**2).sum()
    assert norm_diff < 1e-7, 'The computed homography matrix should equal to the given one.'

    # read the two images as numpy arrays
    im_src = np.array(utils.ReadData(source_image)[0])
    im_ref = np.array(utils.ReadData(ref_image)[0])

    # project source image to the reference image using the homography matrix
    # the size of canvas is specified to store all images after projections.
    im_src_warp = cv2.warpPerspective(im_src, h, (canvas_width, canvas_height))

    # warp_list should contain all images, where the first
    # element is the reference image
    warp_list = [im_ref, im_src_warp]
    result = utils.MergeWarppedImages(canvas_height, canvas_width, warp_list)

    # plot the result of the warping
    plt.figure(figsize=(20, 20))
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()
