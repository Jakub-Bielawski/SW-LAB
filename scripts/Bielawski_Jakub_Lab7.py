from __future__ import print_function
import numpy as np
import cv2 as cv
import time

# from matplotlib import pyplot as plt


def FAST():
    img = cv.imread('forward-1.bmp', cv.IMREAD_COLOR)
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    start = time.time()
    kp = fast.detect(img, None)
    stop = time.time()
    print("Time: ", stop - start, " s")
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    cv.imwrite('fast_true.png', img2)
    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imwrite('fast_false.png', img3)


def ORB():
    img = cv.imread('forward-1.bmp', cv.IMREAD_COLOR)
    # Initiate FAST object with default values
    orb = cv.ORB_create()
    # find and draw the keypoints
    start = time.time()
    kp = orb.detect(img, None)
    stop = time.time()
    print("Time: ", stop - start, " s")
    kp, des = orb.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imwrite('orb_true.png', img2)




def ex_1():
    img_1 = cv.imread('forward-1.bmp', cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread('forward-2.bmp', cv.IMREAD_GRAYSCALE)
    detector = cv.ORB_create()

    detector = cv.AKAZE_create()
    descriptor = cv.xfeatures2d.BriefDescriptorExtractor_create()
    # descriptor = cv.OR

    kp1 = detector.detect(img_1,None)
    kp2 = detector.detect(img_2,None)

    kp1 ,des1=descriptor.compute(img_1,kp1)
    kp2 ,des2=descriptor.compute(img_2,kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches =sorted( matches,key=lambda x: x.distance)
    image3 = cv.drawMatches(img_1,kp1,img_2,kp2,matches,None)
    cv.imwrite("matches.jpg",image3)
def panorama():
    left=cv.imread('left.jpg',cv.IMREAD_GRAYSCALE)
    right=cv.imread('right.jpg',cv.IMREAD_GRAYSCALE)
    detector = cv.AKAZE_create()
    # descriptor = cv.xfeatures2d.BriefDescriptorExtractor_create()
    descriptor=cv.AKAZE_create()
    kp1 = detector.detect(left,None)
    kp2 = detector.detect(right,None)

    kp1 ,des1=descriptor.compute(left,kp1)
    kp2 ,des2=descriptor.compute(right,kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches =sorted( matches,key=lambda x: x.distance)
    image3 = cv.drawMatches(left,kp1,right,kp2,matches,None)
    cv.imwrite("Panorama.jpg",image3)




def example():
    # !/usr/bin/env python

    '''
    Stitching sample
    ================

    Show how to use Stitcher API from python in a simple way to stitch panoramas
    or scans.
    '''

    # Python 2/3 compatibility


    import numpy as np


    import argparse
    import sys

    modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

    parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
    parser.add_argument('--mode',
                        type=int, choices=modes, default=cv.Stitcher_PANORAMA,
                        help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                             'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                             'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--output', default='result.jpg',
                        help='Resulting image. The default is `result.jpg`.')
    parser.add_argument('img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()

    def main():
        args = parser.parse_args()

        # read input images
        imgs = []
        for img_name in args.img:
            img = cv.imread(cv.samples.findFile(img_name))
            if img is None:
                print("can't read image " + img_name)
                sys.exit(-1)
            imgs.append(img)

        stitcher = cv.Stitcher.create(args.mode)
        status, pano = stitcher.stitch(imgs)

        if status != cv.Stitcher_OK:
            print("Can't stitch images, error code = %d" % status)
            sys.exit(-1)

        cv.imwrite(args.output, pano)
        print("stitching completed successfully. %s saved!" % args.output)

        print('Done')

    if __name__ == '__main__':
        print(__doc__)
        main()
        cv.destroyAllWindows()


def main():
    # FAST()
    # ORB()
    # ex_1()
    # panorama()
    example()

if __name__ == "__main__":
    main()
