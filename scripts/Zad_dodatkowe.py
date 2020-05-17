import numpy as np
import cv2


def Zad1():
    """
    Obliczyć splot filtru danego równaniem poniżej z obrazem źródłowym w skali szarości.
    Z wyniku wyciągnąć wartość bezwzględną, wyskalować do zakresu wartości od 0 do 255 i wyświetlić.

        M=np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
    """
    print("Zad1")
    k = 4
    M = np.array([[0, -k / 4, 0],
                  [-k / 4, k, -k / 4],
                  [0, -k / 4, 0]])

    cv2.namedWindow('image')
    image = cv2.imread('shapes.jpg', cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_filtred = (cv2.filter2D(image_gray, cv2.CV_64F, M))
    abs_image = np.absolute(image_filtred)
    image_scaled = np.uint8(abs_image)
    image_scaled = cv2.resize(image_scaled, dsize=(500, 500), dst=0)
    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', image_scaled)
    cv2.destroyAllWindows()


def Zad2():
    """
    W dostarczonym obrazie zakodowano napis na najmłodszym bicie. Wykonać operacje modyfikujące obraz w sposób
    umożliwiające odczytanie zakodowanego napisu.
    """
    cv2.namedWindow('image')
    image = cv2.imread('zad_bit.png', cv2.IMREAD_GRAYSCALE)
    nowy = np.zeros((image.shape[0],image.shape[1]))
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            temp = np.uint8(image[i, j]) & 1
            if temp == 1:
                nowy[i, j] = 255
            else:
                nowy[i, j] = 0
    image_scaled1 = cv2.resize(nowy,dsize=(0, 0), dst=0,fx=0.5,fy=0.5)
    image_scaled2 = cv2.resize(image, dsize=(0, 0), dst=0,fx=0.5,fy=0.5)

    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image', image_scaled1)
        cv2.imshow('image1', image_scaled2)


def main():
    print("Start")
    # Zad1()
    # Zad2()


if __name__ == "__main__":
    main()
