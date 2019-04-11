import math
import numpy
import numpy as np
import init_centroids
from scipy.misc import imread
import matplotlib.pyplot as plt


class Centroid:
    _location = None
    _assigned_pixels = []

    def __init__(self, init_location):
        self._location = init_location
        self._assigned_pixels = []

    def get_location(self):
        return self._location

    def get_floor_location(self):
        floor_location = [0, 0, 0]
        # floor the location
        floor_location[0] = numpy.floor(self._location[0] * 100) / 100
        floor_location[1] = numpy.floor(self._location[1] * 100) / 100
        floor_location[2] = numpy.floor(self._location[2] * 100) / 100
        return floor_location

    def assign_pixel(self, pixel):
        self._assigned_pixels.append(pixel)

    def clear_pixels(self):
        self._assigned_pixels = []

    def update_location(self):
        new_location = [0, 0, 0]
        size = self._assigned_pixels.__len__()
        # Update centroid to be the average of the points in its cluster
        for pixel in self._assigned_pixels:
            new_location[0] = new_location[0] + pixel[0]
            new_location[1] = new_location[1] + pixel[1]
            new_location[2] = new_location[2] + pixel[2]

        if size != 0:
            new_location[0] /= size
            new_location[1] /= size
            new_location[2] /= size

        # update the centroid location
        self._location = new_location

    def print_cent(self):
        cent = self.get_location()
        if type(cent) == list:
            cent = np.asarray(cent)
        if len(cent.shape) == 1:
            return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                                   ']').replace(
                ' ', ', ')
        else:
            return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                                   ']').replace(
                ' ', ', ')[1:-1]


def distance(x1, x2):
    return math.sqrt(pow(x1[0]-x2[0], 2)+pow(x1[1]-x2[1], 2)+pow(x1[2]-x2[2], 2))


def print_centroids_locations(centroids):
    first = True
    for cent in centroids:
        if first:
            print(" ", end='')
            first = False
        else:
            print(", ", end='')
        location_to_print = cent.print_cent()
        print(location_to_print, end='')
    print(flush=True)


def display_image(X, centroids, img_size):

        def calc_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

        array = [c.get_location() for c in centroids]
        B = []
        # create the new picture array
        for pixel in X:
            smallestDist = calc_distance(pixel, array[0])
            smallestIndex = 0
            index = 0
            # check witch centroid is the closest to the current pixel
            for centroid in array:
                dist = calc_distance(pixel, centroid)
                if smallestDist > dist:
                    smallestDist = dist
                    smallestIndex = index
                index += 1
            B.append(array[smallestIndex])
        B = np.array(B)
        # plot the image
        B = B * 255
        B = B.astype(int)
        Y = B.reshape(img_size[0], img_size[1], img_size[2])
        plt.imshow(Y)
        plt.grid(False)
        plt.show()


def main():
    # data preparation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])

    # run over all the K values
    for i in range(1, 5):
        # the K value
        K = pow(2, i)
        # K centroids that are to be used in K-Means on the data set X
        initial_centroids = init_centroids.init_centroids(X, K)

        centroids = []
        for cent in initial_centroids:
            centroids.append(Centroid(cent))

        print("k=" + K.__str__() + ":")
        print("iter 0 :", end='')
        print_centroids_locations(centroids)

        for j in range(1, 11):
            for pixel in X:
                min_dist = distance(pixel, centroids[0].get_location())
                new_cent = centroids[0]
                for cent in centroids:
                    dist = distance(pixel, cent.get_location())
                    if dist < min_dist:
                        min_dist = dist
                        new_cent = cent
                new_cent.assign_pixel(pixel)

            # update all the centroids location
            for cent in centroids:
                cent.update_location()

            # print centroids location
            print("Iter", j, ":", end=" ")
            print_centroids_locations(centroids)

            # clear all the assigned pixels
            for cent in centroids:
                cent.clear_pixels()

        # displays photo at the end of iteration
        display_image(X, centroids, img_size)


main()

