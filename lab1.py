import numpy as np
import cv2


def print(name, input, output):
    result = np.hstack((input, output))
    cv2.imshow(name, result)


def main():
    # 1) Прочитать изображение из файла
    image = cv2.imread('image.jpg')
    height, width, channels = image.shape

    # 2) Исходное цветное изображение преобразовать в полутоновое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3_channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    print("gray", image, gray_3_channel)

    # 3) Улучшить контраст
    contrast_image = cv2.equalizeHist(gray_image)
    print("contrast", gray_image, contrast_image)

    # 4) Найти края объектов методом Canny
    canny_image = cv2.Canny(gray_image, 50, 150)
    print("canny", gray_image, canny_image)

    # 5) Найти угловые точки на изображении. Нарисовать их кругом с радиусом r = 2 в тоже изображение, где края
    corner_image = cv2.cornerHarris(canny_image, 2, 3, 0.1)

    corner_image[corner_image > (0.01 * corner_image.max())] = 255

    circled_corner = np.zeros(gray_image.shape, dtype="ubyte")
    for j in range(width):
        for i in range(height):
            if corner_image[i, j] > 250:
                cv2.circle(circled_corner, (j, i), 2, (255, 0, 0), -1)

    corner_image = canny_image + circled_corner
    print("corners", canny_image, corner_image)

    # 6) Для найденных границ и угловых точек строится карта расстояний D[i,j] методом distance transform
    distance_image = cv2.distanceTransform(255 - corner_image, cv2.DIST_L2, 3)
    cv2.normalize(distance_image, distance_image, 0, 1.0, cv2.NORM_MINMAX)
    print("distance_transform", corner_image, distance_image)

    # 7) В каждом пикселе [i,j] производится фильтрация усреднением.
    # Размер фильтра для усреднения зависит от расстояния до угловых
    # и краевых точек и равен k*D[i,j], где к – параметр алгоритма.
    # Таким образом, чем дальше от края или угла, тем больше усреднение (больше подавление шума)

    # 8) Для ускорения вычислений при усреднении нужно использовать интегральные изображения.

    integral = cv2.integral(image)
    result = np.zeros(image.shape, dtype="ubyte")
    xs = np.arange(image.shape[0], dtype='int32')
    ys = np.arange(image.shape[1], dtype='int32')
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    k = 70
    rad = (k * distance_image).astype('int32') + 1

    x_left = np.maximum(X - rad, 0)
    x_right = np.minimum(X + rad, image.shape[0] - 1)
    y_left = np.maximum(Y - rad, 0)
    y_right = np.minimum(Y + rad, image.shape[1] - 1)

    for i in range(height):
        for j in range (width):
            for ch in range(channels):
                x_l = x_left[i][j]
                x_r = x_right[i][j]
                y_l = y_left[i][j]
                y_r = y_right[i][j]

                A = integral[x_l][y_l][ch]
                B = integral[x_r][y_l][ch]
                C = integral[x_l][y_r][ch]
                D = integral[x_r][y_r][ch]

                result[i][j][ch] = (D + A - B - C) / ((x_r - x_l) * (y_r - y_l))

    print("averaging", image, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()