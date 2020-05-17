import cv2
import numpy as np
import os
import glob


def readFiles(path,ext):
    img_dir = path   # Enter Directory of all images
    if ext == "png":
        ext='png'
    elif ext == "bmp":
        ext = 'bmp'
    data_path = img_dir + '/*.'+ext
    files = glob.glob(data_path)
    #
    data = [cv2.imread(f1) for f1 in files]
    print("Found ", len(data), " files")
    return data
def robFocisze():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0
    print("Hit SPACE to take photo ")
    print("")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "/home/jakub/PycharmProjects/SW-LAB/cameraCalibration/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def zad_1(data):
    image_points = []
    object_points_all = []
    flags = []
    for image in data:
        # image = cv2.imread('calib_basler21130751/img_21130751_0000.bmp')
        flag_found, corners = cv2.findChessboardCorners(image, (8, 5))

        flags.append(flag_found)
        # print(corners)
        if flag_found:
            image_points.append(corners)
            print('Corners found')
            corners = cv2.cornerSubPix(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1)
            )
            # print(corners[0])
            object_points = np.zeros((8 * 5, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
            # print(object_points[0:3])
            object_points_all.append(object_points)
        else:
            print("Corners not found")
    print("Processing...")
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_all, image_points, data[0].shape[:2], None, None)
    fovx, fovy, focallLength, principalPoints, aspectRatio = cv2.calibrationMatrixValues(
        camera_matrix, data[0].shape[0:2], 7.2, 5.4
    )
    print("Fovx: ",fovx)
    print("Fovy: ",fovy)
    print("Focall Length: ",focallLength)

    image_with_corners = cv2.drawChessboardCorners(data[0], (8, 5), image_points[0], flags[0])
    img_undistored = cv2.undistort(data[0], camera_matrix, dist_coeffs)
    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
        cv2.imshow('image_with_conrners', image_with_corners)
        cv2.imshow('image_undistored', img_undistored)

    cv2.destroyAllWindows()


def main():
    robFocisze()
    path = input("Give my directory with images.\n")
    extension = input("Give me extension of images.\n")
    data = readFiles(path=path, ext=extension)
    zad_1(data)


if __name__ == '__main__':
    main()
