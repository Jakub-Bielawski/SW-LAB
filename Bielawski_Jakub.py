import cv2
import ds as ds
import numpy as np
import os
from time import perf_counter

def empty_callback(value):
    print(f'Trackbar reporting for duty with value: {value}')

# create a black image, a window


def zad1():
    img = cv2.imread('gory_1.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('image')

    cv2.createTrackbar('T_V', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('Rodzaj progowania', 'image', 0, 7, empty_callback)


    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        t_v = cv2.getTrackbarPos('T_V', 'image')
        value = cv2.getTrackbarPos('Rodzaj progowania', 'image')
        if value == 0:
            rodzaj = cv2.THRESH_BINARY
        elif value == 1:
            rodzaj = cv2.THRESH_BINARY_INV
        elif value == 2:
            rodzaj = cv2.THRESH_MASK
        elif value == 3:
            rodzaj = cv2.THRESH_OTSU
        elif value == 4:
            rodzaj = cv2.THRESH_TOZERO
        elif value == 5:
            rodzaj = cv2.THRESH_TOZERO_INV
        elif value == 6:
            rodzaj = cv2.THRESH_TRIANGLE
        elif value == 7:
            rodzaj = cv2.THRESH_TRUNC

        ret, thresh = cv2.threshold(img, t_v, 255, rodzaj)
        cv2.imshow('image', thresh)


    ################################################################
    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()



def zad2():
    print('zad2')
    img = cv2.imread('qr.jpg', cv2.IMREAD_COLOR)
    # cv2.namedWindow('image0')
    # cv2.namedWindow('image1')
    # cv2.namedWindow('image2')
    # cv2.namedWindow('image3')
    czas_trwania = 0
    czas_start = 0
    czas_stop = 0
    for i in range(4):
        cv2.namedWindow(f'image{i}')
    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
        img_list = []

        czas_start=perf_counter()
        temp=cv2.resize(img, dsize=(0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_LINEAR)
        czas_stop=perf_counter()
        if czas_trwania==0:
            czas_trwania=czas_stop-czas_start
            print('Czas trwanaia skalowania :', czas_trwania*1000, 'ms')
        img_list.append(temp)
        img_list.append(cv2.resize(img, dsize=(0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_NEAREST))
        img_list.append(cv2.resize(img, dsize=(0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_AREA))
        img_list.append(cv2.resize(img, dsize=(0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_LANCZOS4))

        for i in range(4):
            cv2.imshow(f'image{i}', img_list[i])

    cv2.destroyAllWindows()



def zad3():
    ilosc_obrazow=1
    print("zad3")
    img_1=cv2.imread("LOGO_PUT_VISION_LAB_MAIN.png")
    img_2=cv2.imread("qr.jpg")
    height1, width1, channels1 = img_1.shape
    height2, width2, channels2 = img_2.shape
    scale=height1/height2
    print(scale)

    for i in range(ilosc_obrazow):
        cv2.namedWindow(f'image{i}')

    cv2.createTrackbar('a-b', 'image0', 0, 100, empty_callback)

    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        img_21=cv2.resize(img_2, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        value = cv2.getTrackbarPos('a-b', 'image0')/100
        dst = cv2.addWeighted(img_1, value, img_21, 1-value, 0)
        cv2.imshow('image0', dst)



    cv2.destroyAllWindows()


def zad_negatyw(img):
    img_copy = img
    rows, cols, channels = img.shape
    print(channels)
    print('Tworzenie negatywu')
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                img_copy[i, j, k] = 255-img[i, j, k]
    return img_copy


def negatyw():
    img = cv2.imread('gory_1.jpg', cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    negatyw=zad_negatyw(img)

    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
        cv2.imshow('image', negatyw)

    cv2.destroyAllWindows()
def main():

    # zad1()
    # input("")  # wcisnac ENTER, nie mam pojecia jak zrobic funkcje ktora wykrywa kazdy klawisz
    zad2()
    # input("")
    # zad3()
    # input("")
    # negatyw()


if __name__ == '__main__':
    main()


