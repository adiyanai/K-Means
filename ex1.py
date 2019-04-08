import math
import numpy
import init_centroids
from scipy.misc import imread


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
        location_to_print = cent.get_floor_location()
        print('[{0}, {1}, {2}]'.format(location_to_print[0], location_to_print[1], location_to_print[2]), end='')
    print(flush=True)


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


main()

