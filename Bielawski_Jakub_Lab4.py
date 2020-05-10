import cv2
import ds as ds
import numpy as np
import os
from matplotlib import pyplot as plt
from time import perf_counter

# mouse callback function


global i
i = 0


def draw_circle(event, x, y, flags, img):
    global i
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x - 10, y - 10)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        cv2.putText(img, f'num{i}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        i += 1
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img, (x + 20, y + 20), (x - 20, y - 20), (0, 255, 0), -1)
        cv2.putText(img, f'num{i}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        i += 1


# Create a black image, a window and bind the function to window


def Myszka():
    cv2.namedWindow('image')
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.setMouseCallback('image', draw_circle, img)

    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


lista_krokow = []
road = cv2.imread("road.jpg", cv2.IMREAD_COLOR)


def zaznaczanie(event, x, y, flags, parama):
    print(len(lista_krokow))
    if not len(lista_krokow) == 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            temp = [x, y]
            lista_krokow.append(temp)


def Transformacje():
    dst = road
    rows, columns, channels = road.shape
    cv2.namedWindow('image')
    cv2.namedWindow('clone')
    cv2.setMouseCallback('image', zaznaczanie)

    while (1):
        cv2.imshow('image', road)

        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(lista_krokow) == 4:
            max_x = max(lista_krokow)[0]
            max_y = max(lista_krokow)[1]

            pts1 = np.float32([lista_krokow[0], lista_krokow[1], lista_krokow[2],
                               lista_krokow[3]])
            pts2 = np.float32([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(road, M, (max_x, max_y))
            cv2.imshow('clon', dst)
    cv2.destroyAllWindows()


def Histogramy():
    img = cv2.imread("gory_5.jpg", cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread("gory_5.jpg", cv2.IMREAD_COLOR)
    img_2 = cv2.imread('clahe.jpg')
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_1], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

    clahe = cv2.createCLAHE(2.0, (8, 8))
    cl1 = clahe.apply(img)
    cv2.imwrite('grayscale.jpg', img)
    cv2.imwrite('clahe.jpg', cl1)


Punkty = []


def Zaznacz(event, x, y, flags, parama=2):
    if not len(Punkty) == parama:
        if event == cv2.EVENT_LBUTTONDOWN:
            temp = [x, y]
            Punkty.append(temp)


def Zad3():
    img = cv2.imread("gory_5.jpg", cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', Zaznacz)
    g = img[:, :, 1]

    while (1):

        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(Punkty) == 2:

            fragment = g[Punkty[0][1]:Punkty[1][1], Punkty[0][0]:Punkty[1][0]]
            ret, thresh = cv2.threshold(fragment, 123, 255, cv2.THRESH_BINARY)
            img[Punkty[0][1]:Punkty[1][1], Punkty[0][0]:Punkty[1][0], 1] = thresh
            cv2.imshow('image', img)
            Punkty.clear()
        else:

            cv2.imshow('image', img)
    cv2.destroyAllWindows()


def Mopsik():
    print("MOPS")
    image = cv2.imread('gallery.png', cv2.IMREAD_COLOR)
    mops = cv2.imread('pug.png', cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    cv2.namedWindow('bg')
    cv2.namedWindow('fg')

    klikniecia = 4
    # Punkty = []
    cv2.setMouseCallback('image', Zaznacz, klikniecia)
    pts1 = np.float32([[0, mops.shape[0]], [0, 0], [mops.shape[1], 0], [mops.shape[1], mops.shape[0]]])
    print(pts1)
    while (1):
        print(len(Punkty))
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(Punkty) == klikniecia:
            pts2 = np.float32([Punkty[0], Punkty[1], Punkty[2], Punkty[3]])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            obraz = cv2.warpPerspective(mops, M, (image.shape[1], image.shape[0]))

            roi = image
            img2gray = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 2, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(obraz, obraz, mask=mask)
            dst = cv2.add(img1_bg, img2_fg)
            image=dst
            cv2.imshow('bg',img1_bg)
            cv2.imshow('fg',img2_fg)
            cv2.imshow('image',image)
        else:

            cv2.imshow('image',image)
    cv2.destroyAllWindows()


def main():
    print("Start")
    Myszka()
    input("")
    Transformacje()
    input("")
    Histogramy()
    input("")
    Zad3()
    input("")
    Mopsik()


if __name__ == "__main__":
    main()
