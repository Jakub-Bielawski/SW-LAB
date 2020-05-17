import cv2
import numpy as np
from matplotlib import pyplot as plt


def Zad1():
    print("Zad_1")
    Prewitt_X = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
    Prewitt_Y = np.transpose(Prewitt_X)
    Sobel_X = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    Sobel_Y = np.transpose(Sobel_X)
    cv2.namedWindow('image')
    cv2.namedWindow('image1')
    cv2.namedWindow('image2')

    image = cv2.imread('gallery.png', cv2.IMREAD_GRAYSCALE)

    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
        nowiutki_obraz_x = (cv2.filter2D(image, cv2.CV_64F, Sobel_X))
        nowiutki_obraz_y = (cv2.filter2D(image, cv2.CV_64F, Sobel_Y))
        # nowiutki_obraz_y=(cv2.filter2D(image,-1,Prewitt_Y))
        abs_nowiutki_x = np.absolute(nowiutki_obraz_x)
        abs_nowiutki_y = np.absolute(nowiutki_obraz_y)
        nowiutki_x = np.uint8(abs_nowiutki_x)
        nowiutki_y = np.uint8(abs_nowiutki_y)
        # print(type(nowiutki_obraz_x))
        nowiutki_obraz = nowiutki_x + nowiutki_y
        cv2.imshow('image', nowiutki_x)
        cv2.imshow('image1', nowiutki_y)
        cv2.imshow('image2', nowiutki_obraz)


def Zad2():
    def empty_callback(pos):
        pass

    # Wykrywanie krawedzu metoda Canny'ego
    print("Zad2")
    cv2.namedWindow('image')
    cv2.createTrackbar('T1', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('T2', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('T3', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('G', 'image', 0, 255, empty_callback)
    drone = cv2.imread('drone_ship.jpg')
    cap = cv2.VideoCapture(0)
    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
        ret, frame = cap.read()
        img_gray = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        rozmiar_gausian = cv2.getTrackbarPos('G', 'image')
        rozmiar_gausian = 2 * rozmiar_gausian - 1
        if rozmiar_gausian <= 3:
            rozmiar_gausian = 3
        img_filtered = cv2.GaussianBlur(img_gray, (rozmiar_gausian, rozmiar_gausian), 1.5)
        T1 = cv2.getTrackbarPos('T1', 'image')
        T2 = cv2.getTrackbarPos('T2', 'image')
        T3 = cv2.getTrackbarPos('T3', 'image')

        img_edge = cv2.Canny(img_filtered, T1, T2, T3)
        cv2.imshow('image', img_edge)
    cv2.destroyAllWindows()


def Zad3():
    print("Zad3")
    img = cv2.imread(cv2.samples.findFile('shapes.jpg'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 200)
    print(type(lines))  # 50 20       20 70
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 10, param1=20, param2=70, minRadius=0,
                               maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.namedWindow('image')
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break

        cv2.imshow('image', img)

    cv2.destroyAllWindows()


def Zad4():
    def callback(pos):
        pass

    print("Zad4")
    image = cv2.imread(cv2.samples.findFile('drone_ship.jpg'))
    image = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gausian=cv2.GaussianBlur(gray,(7,7),1.5)
    canny = cv2.Canny(gray, 100, 255, apertureSize=3)
    # lines = cv2.HoughLines(canny, 1, np.pi / 180, 130)
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * a)
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * a)
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, gray.shape[0] / 4, param1=250, param2=140, minRadius=0,
                               maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break

        cv2.imshow('image', image)
    cv2.destroyAllWindows()


def Zad5():
    print("Zad5")
    image = cv2.imread('fruit.jpg')
    cv2.namedWindow('image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, gray.shape[0] / 10, param1=200, param2=140, minRadius=0,
                               maxRadius=0)
    circles = np.uint16(np.around(circles))
    # image_cpy=image
    obrazy = []
    wartosci = []
    for i in circles[0, :]:
        x = i[0]
        y = i[1]
        r = int(0.7 * i[2])
        temp = image[y - r:y + r, x - r:x + r, 2]
        obrazy.append(temp)
        wartosc = np.sum(temp)
        wartosci.append(wartosc)
    max_val = max(wartosci)
    wartosci /= max_val

    for index, i in enumerate(circles[0, :]):
        # draw the outer circle
        if wartosci[index] > 0.8:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 4)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 4)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', image)
    cv2.destroyAllWindows()


def Zad6():
    print("Zad6")
    image = cv2.imread('coins.jpg')
    cv2.namedWindow('image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, gray.shape[0] / 10, param1=380, param2=170, minRadius=0,
                               maxRadius=250)
    circles = np.uint16(np.around(circles))
    kwota = 0.0
    for i in circles[0, :]:
        if i[2] > 70:
            kwota += 1
        else:
            kwota += 0.1

        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)

        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    print("Kwota na obrazie to :",round(kwota,2), "zł.")
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', image)
    cv2.destroyAllWindows()


def main():
    print("Start")
    # Zad1()
    # Zad2()
    # Zad3()
    # Zad4()
    # Zad5()
    Zad6()


if __name__ == "__main__":
    main()
