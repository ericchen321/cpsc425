import numpy as np
import cv2
import math
import random

def ComputeAngleInDeg0to180(angleInRad):
    """
    Given angle in radians in [-inf, inf], returns
    angle in degrees in [0, 180].
    """
    # map to [0, 2*pi]
    angleInRad0to2Pi = angleInRad % (2*math.pi)
    # map to [0, 360]
    angleInDeg0to360 = math.degrees(angleInRad0to2Pi)
    # map to [0, 180]
    angleInDeg0to180 = angleInDeg0to360
    if angleInDeg0to360 > 180.0:
        angleInDeg0to180 = 360.0 - angleInDeg0to360
    return angleInDeg0to180

def ComputeDiffs(pair, keypoints1, keypoints2):
    """
    Computes and returns the orientation and scale difference between
    keypoints1 and keypoints2.
    Inputs:
        pair: a tuple (i, j), indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation. Orientation
            measured in radians
    Output:
        tuple (o_diff, s_diff) for orientation and scale difference.
        o_diff is angle(keypoints2[j]) - angle(keypoints1[i]), measured
        in degrees, in the range [0, 180];
        s_diff is scale(keypoints2[j])/scale(keypoints1[i]).
    """
    # retrieve orientations in radians in the range [-inf, inf]
    angle_i = keypoints1[pair[0], 3]
    angle_j = keypoints2[pair[1], 3]

    # compute angular difference in degrees, in the range [0, 180]
    o_diff = ComputeAngleInDeg0to180(angle_j - angle_i)

    # compute scale difference
    s_diff = keypoints2[pair[1], 2] / keypoints1[pair[0], 2]
    return (o_diff, s_diff)

def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    # if no matched pairs, quit
    if len(matched_pairs) == 0:
        return []

    # initialize largest set
    largest_set = []

    # do 10 times
    for iteration in range(0, 10):
        # randomly draw one pair out
        pair_index = random.randint(0, len(matched_pairs)-1)
        pair = matched_pairs[pair_index]

        # initialize temporary set
        temp_set = [pair]

        # compute orientation and scale difference
        o_diff, s_diff = ComputeDiffs(pair, keypoints1, keypoints2)
        #print(f"o_diff is {o_diff}, ")

        # for all other matched pairs
        for pair_other_index in range(0, len(matched_pairs)):
            if pair_other_index != pair_index:
                # select the pair
                pair_other = matched_pairs[pair_other_index]
                # compute orientation and scale difference
                o_diff_other, s_diff_other = ComputeDiffs(pair_other, keypoints1, keypoints2)
                # check consistency. If consistent, add to temp set
                if abs(o_diff_other - o_diff)<=orient_agreement and s_diff_other<=s_diff*(1.0+scale_agreement) and s_diff_other>=s_diff*(1.0-scale_agreement): 
                    temp_set.append(pair_other)
        
        print(f"temp set size: {len(temp_set)}")
        # replace largest set so far with the temp set,
        # if the temp set is larger
        if len(temp_set)>len(largest_set):
            largest_set = temp_set
    print(f"largest set size: {len(largest_set)}")
    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    # produce candidate matches
    # get cos values
    cosValues = np.matmul(descriptors1, np.transpose(descriptors2))
    # get acos values of shape (K1, K2)
    simScores = np.arccos(cosValues)
    # initialize array of shape (K1, ) to represent matched pairs
    matchedPairsArr = np.argmin(simScores, axis=1)

    # eliminate false matches
    # sort simScores - index no longer corresponds to original order
    simScoresSorted = np.sort(simScores, axis=1)
    # produce a binary array of shape (K1, ) to represent if a match
    # is above the threshold
    assert simScoresSorted.shape[1] >= 2
    ifAboveThresh = simScoresSorted[:, 0]/simScoresSorted[:, 1] > threshold
    # mask the matched pair array to mark false matches
    matchedPairsArr[ifAboveThresh] = -1

    # resolve conflicts
    # TODO:    

    # assemble pairs and remove false matches
    matched_pairs = []
    matchedPairsWithFalse = [(i, matchedPairsArr[i]) for i in range(0, matchedPairsArr.shape[0])]
    for pair in matchedPairsWithFalse:
        if pair[1] != -1:
            matched_pairs.append(pair)
    print(f"number of matched pairs: {len(matched_pairs)}")
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # convert to homogeneous coordinates
    xy_points_homo_source = np.append(xy_points, np.ones((xy_points.shape[0], 1), dtype=np.float64), axis=1) # (num_points, 3)
    
    # project
    xy_points_homo_proj = np.transpose(np.matmul(h, np.transpose(xy_points_homo_source))) # (num_points, 3)
    
    # get the z componenets of projected points
    z = xy_points_homo_proj[:, -1]
    z[z == 0.0] = 1e-10 # (num_points, )
    z = np.repeat(np.expand_dims(z, axis=1), 2, axis=1) # (num_points, 2)
    
    # reduce to non-homo coordinates
    xy_points_out = xy_points_homo_proj[:, 0:-1] / z
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START



    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
