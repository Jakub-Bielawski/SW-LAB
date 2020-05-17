import cv2
import numpy as np


def Zad1():
    print("Zad1")
    cv2.namedWindow('image')
    image = cv2.imread('not_bad.jpg', cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    kernel1 = np.ones((9, 9), np.uint8)

    image_erode = cv2.erode(thresh, kernel)
    image_dilation = cv2.dilate(image_erode, kernel1)
    image_contours, image_hierarchy = cv2.findContours(image_dilation, mode=cv2.RETR_TREE,
                                                       method=cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, image_contours, 0, (0, 0, 0), 3)
    cv2.drawContours(image, image_contours, 1, (255, 0, 0), 3)
    cv2.drawContours(image, image_contours, 2, (0, 255, 0), 3)
    cv2.drawContours(image, image_contours, 3, (0, 0, 255), 3)
    cv2.drawContours(image, image_contours, 4, (255, 255, 0), 3)

    image_centers = []

    for cnt in image_contours:
        M = cv2.moments(cnt)
        print(cv2.contourArea(cnt))
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        point = [cx, cy]

        # size=5
        # image[cy-size:cy+size, cx-size:cx+size] = [255 ,255, 255]
        if cv2.contourArea(cnt) < 100000:
            image_centers.append(point)

    l_d = image_centers[1]
    l_g = image_centers[3]
    p_g = image_centers[2]
    p_d = image_centers[0]
    max_x = 1000
    max_y = 1000

    pts1 = np.float32([l_g, p_g, p_d, l_d])

    pts2 = np.float32([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (max_x, max_y))

    image_resized = cv2.resize(dst, dsize=(0, 0), dst=0, fx=0.3, fy=0.3)
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', image_resized)

    cv2.destroyAllWindows()


def Zad2():
    image = cv2.imread('pug.png', cv2.IMREAD_COLOR)
    img2 = image.copy()
    template = image[17:110, 85:188]
    cv2.namedWindow('image')
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    img = img2.copy()
    method = 0
    res = cv2.matchTemplate(img, template, method)
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', res)
    cv2.destroyAllWindows()


def main():
    print("Start")
    # Zad1()
    # Zad2()


if __name__ == "__main__":
    main()
