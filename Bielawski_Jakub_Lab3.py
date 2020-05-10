import cv2
import ds as ds
import numpy as np
import os
from matplotlib import pyplot as plt
from time import perf_counter
from tqdm import tqdm


def on_change(value):
    pass


def Wyswietl(img):
    cv2.namedWindow('image')
    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
        cv2.imshow('image', img)

    cv2.destroyAllWindows()


def empty_callback(value):
    pass


def Filtracja(img):
    print("Filtracja")

    cv2.namedWindow('image')
    cv2.namedWindow('image blur')
    cv2.namedWindow('image gausian')
    cv2.namedWindow('image median')

    cv2.createTrackbar('rozmiar g', 'image gausian', 3, 99, on_change)
    cv2.createTrackbar('rozmiar b', 'image blur', 3, 99, on_change)
    cv2.createTrackbar('rozmiar m', 'image median', 3, 99, on_change)
    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
        rozmiar_blur = cv2.getTrackbarPos('rozmiar b', 'image blur')
        rozmiar_gausian = cv2.getTrackbarPos('rozmiar g', 'image gausian')
        rozmiar_medina = cv2.getTrackbarPos('rozmiar m', 'image median')

        rozmiar_blur = 2 * rozmiar_blur - 1
        rozmiar_gausian = 2 * rozmiar_gausian - 1
        rozmiar_medina = 2 * rozmiar_medina - 1
        if (rozmiar_medina <= 3):
            rozmiar_medina = 3
        if rozmiar_gausian <= 3:
            rozmiar_gausian = 3
        if rozmiar_blur <= 3:
            rozmiar_blur = 3
        img_blur = cv2.blur(img, (rozmiar_blur, rozmiar_blur))
        img_gausian = cv2.GaussianBlur(img, (rozmiar_gausian, rozmiar_gausian), 0)
        img_median = cv2.medianBlur(img, rozmiar_medina)

        cv2.imshow('image', img)
        cv2.imshow('image blur', img_blur)
        cv2.imshow('image gausian', img_gausian)
        cv2.imshow('image median', img_median)

    cv2.destroyAllWindows()


def Progowanie(img, value, wartosc_progu):
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

    ret, thresh = cv2.threshold(img, wartosc_progu, 255, rodzaj)

    return thresh


def Morfologia(img):
    print("Morfologia")
    rodzaj_progowania = 0
    prog = 127
    obraz = Progowanie(img, rodzaj_progowania, prog)
    cv2.namedWindow('Wzor')
    cv2.namedWindow('Erosion')
    cv2.namedWindow('Dilation')
    cv2.namedWindow('Opening')
    cv2.namedWindow('Closing')

    cv2.createTrackbar('Erosion', 'Erosion', 3, 99, on_change)
    cv2.createTrackbar('Dilation', 'Dilation', 3, 99, on_change)
    cv2.createTrackbar('Opening', 'Opening', 3, 99, on_change)
    cv2.createTrackbar('Closing', 'Closing', 3, 99, on_change)
    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        T_erosion = cv2.getTrackbarPos('Erosion', 'Erosion')
        T_dilation = cv2.getTrackbarPos('Dilation', 'Dilation')
        T_opening = cv2.getTrackbarPos('Opening', 'Opening')
        T_closing = cv2.getTrackbarPos('Closing', 'Closing')

        T_dilation = 2 * T_dilation - 1
        T_erosion = 2 * T_erosion - 1
        T_opening = 2 * T_opening - 1
        T_closing = 2 * T_closing - 1

        if T_erosion <= 3:
            T_erosion = 3
        if T_dilation <= 3:
            T_dilation = 3
        if T_opening <= 3:
            T_opening = 3
        if T_closing <= 3:
            T_closing = 3

        kernel_dilation = np.ones((T_dilation, T_dilation), np.uint8)
        kernel_erosion = np.ones((T_erosion, T_erosion), np.uint8)
        kernel_opening = np.ones((T_opening, T_opening), np.uint8)
        kernel_closing = np.ones((T_closing, T_closing), np.uint8)
        img_erosion = cv2.erode(obraz, kernel_erosion, iterations=1)
        img_dilation = cv2.dilate(obraz, kernel_dilation, iterations=1)
        img_opening = cv2.morphologyEx(obraz, cv2.MORPH_OPEN, kernel_opening)
        img_closing = cv2.morphologyEx(obraz, cv2.MORPH_CLOSE, kernel_closing)
        cv2.imshow('Wzor', obraz)
        cv2.imshow('Erosion', img_erosion)
        cv2.imshow('Dilation', img_dilation)
        cv2.imshow('Opening', img_opening)
        cv2.imshow('Closing', img_closing)

    cv2.destroyAllWindows()


def SkanowanieObrazu(img):
    print("Skanowanie obrazu")
    img_height, img_width = img.shape
    print(img.shape)
    print(img_height, img_width)
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if not j % 3:
                # print(i,j)
                img[i, j] = 255
            else:
                pass
    return img


def Wygladzanie(img):
    img_height, im_width = img.shape
    print(im_width, img_height)
    img_copy = img
    temp = 0
    for y in range(1, img_height - 2):
        for x in range(1, im_width - 2):
            for y_p in range(y - 1, y + 2):
                for x_p in range(x - 1, x + 2):
                    # print("x=",x,"y=",y,"x_p=",x_p,"y_p",y_p)
                    temp += img[y_p, x_p]
            img_copy[y, x] = (1 / 9) * temp
            temp = 0

    return img_copy


def PorownanieCzasow(scaned):
    print("Porównywanie czasów")
    czas_start = perf_counter()
    wygladzony = Wygladzanie(scaned)
    czas_stop = perf_counter()
    czas_trwania = czas_stop - czas_start
    print(czas_trwania)

    czas_start_1 = perf_counter()
    img_blur = cv2.blur(scaned, (3, 3))
    czas_stop_1 = perf_counter()
    czas_trwania1 = czas_stop_1 - czas_start_1
    print(czas_trwania1)

    kernel = np.ones((3, 3), np.uint8) / 9
    czas_start_2 = perf_counter()
    filtered2d = cv2.filter2D(scaned, -1, kernel)
    czas_stop_2 = perf_counter()
    czas_trwania2 = czas_stop_2 - czas_start_2
    print(czas_trwania2)
    Wyswietl(wygladzony)
    Wyswietl(img_blur)
    Wyswietl(filtered2d)


def KuwaharaFilter(img, L):
    print("Filtrowanie filtrem Kuwahary")
    kopia = np.array(img)
    result = np.array(img)

    try:
        img_height, img_width, chanells = img.shape
        print(img_height, img_width, chanells)
        kolor = 1
    except ValueError:
        img_width, img_height = img.shape
        print(img_height, img_width)
        chanells = 0
        kolor = 0

    if kolor:
        print("Processing color image")
        print("Image size:", img_width, "x", img_height)
        for k in range(0, chanells):
            print("")
            print(f"Dealing with {k + 1}/3 color.")
            for x in tqdm(range(L, img_height - L)):
                for y in range(L, img_width - L):
                    mean1, std1 = cv2.meanStdDev(kopia[x - L:x + 1, y - L:y + 1, k])
                    mean2, std2 = cv2.meanStdDev(kopia[x - L:x + 1, y - 1:y + L, k])
                    mean3, std3 = cv2.meanStdDev(kopia[x - 1:x + L, y - L:y + 1, k])
                    mean4, std4 = cv2.meanStdDev(kopia[x - 1:x + L, y - 1:y + L, k])

                    slownik = {float(std1): float(mean1),
                               float(std2): float(mean2),
                               float(std3): float(mean3),
                               float(std4): float(mean4)}
                    minimum = min(float(std1), float(std2), float(std3), float(std4))
                    result[x, y, k] = slownik[minimum]
    else:
        for x in tqdm(range(L, img_width - L)):
            for y in range(L, img_height - L):
                mean1, std1 = cv2.meanStdDev(kopia[x - L:x + 1, y - L:y + 1])
                mean2, std2 = cv2.meanStdDev(kopia[x - L:x + 1, y - 1:y + L])
                mean3, std3 = cv2.meanStdDev(kopia[x - 1:x + L, y - L:y + 1])
                mean4, std4 = cv2.meanStdDev(kopia[x - 1:x + L, y - 1:y + L])

                slownik = {float(std1): float(mean1),
                           float(std2): float(mean2),
                           float(std3): float(mean3),
                           float(std4): float(mean4)}
                minimum = min(float(std1), float(std2), float(std3), float(std4))
                result[x, y] = slownik[minimum]
    return result


def main():
    img = cv2.imread('lena_noise.bmp', cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread('lena_salt_and_pepper.bmp')
    img_2 = cv2.imread('SummitwithHail.jpg', cv2.IMREAD_COLOR)
    L_Kuwahara = 7
    # Filtracja(img)
    # input("")
    # Filtracja(img_1)
    # input("")
    # Morfologia(img_2)
    # input("")
    # scaned = SkanowanieObrazu(img_2)
    # # Wyswietl(scaned)
    # PorownanieCzasow(scaned)
    # input("")

    for i in range(42):
        if i % 2:
            print(f"-------------------- {i} -------------")
            Kawuhara = KuwaharaFilter(img_2, i)
            cv2.imwrite(f'pic{i}.jpg', Kawuhara, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    main()
