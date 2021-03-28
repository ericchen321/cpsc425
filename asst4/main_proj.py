from PIL import Image
import numpy as np
import pickle as pkl
import sys
import hw_utils as utils
import solution as sol


def main():
    with open('./data/test.pkl', 'rb') as f:
        test_dict = pkl.load(f)

    # visualize 30 random matches
    num_pts = 30
    idx = np.random.permutation(test_dict['xy_src'].shape[0])[:num_pts]
    xy_src = test_dict['xy_src'][idx]
    xy_ref = test_dict['xy_ref'][idx]
    h = test_dict['h']

    # project the src keypoints to the reference frame using homography
    xy_proj = sol.KeypointProjection(xy_src, h)

    # visualize the results
    im_ref = np.array(Image.open('./data/Hanging1.png'))
    im_src = np.array(Image.open('./data/Hanging2.png'))
    utils.VisualizePointProj(xy_src, xy_ref, xy_proj, im_src, im_ref)

if __name__ == '__main__':
    main()
