from fastai.conv_learner import ConvLearner
from fastai.dataset import ImageClassifierData, tfms_from_model, open_image
from fastai.torch_imports import resnext101_64
from flask import Flask, request
from collections import defaultdict
from functools import partial
import scipy.spatial as spatial
import scipy.cluster as clstr
import numpy as np
import chess.uci
import cv2
import imutils

sz = 227
arch = resnext101_64
app = Flask(__name__)


class Stockfish:
    def __init__(self, path):
        self.info_handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine(path)
        self.engine.uci()
        self.engine.info_handlers.append(self.info_handler)

    def get_best_move(self):
        bestmove, ponder = self.engine.go(movetime=2000)
        return bestmove, self.info_handler.info["score"][1]

    def set_fen_position(self, fen):
        self.engine.position(chess.Board(fen=fen))


def init_models():
    data_empty = ImageClassifierData.from_paths('assets/models/empty', tfms=tfms_from_model(arch, sz))
    learn_empty = ConvLearner.pretrained(arch, data_empty, precompute=False)
    learn_empty.load('empty')

    data_color = ImageClassifierData.from_paths('assets/models/color', tfms=tfms_from_model(arch, sz))
    learn_color = ConvLearner.pretrained(arch, data_color, precompute=False)
    learn_color.load('color')

    data_type = ImageClassifierData.from_paths('assets/models/type', tfms=tfms_from_model(arch, sz))
    learn_type = ConvLearner.pretrained(arch, data_type, precompute=False)
    learn_type.load('type')

    return learn_empty, learn_color, learn_type


def init_stockfish():
    return Stockfish('assets/engine/stockfish_10_x64')


def get_fen_char(pred_type, pred_color):
    if pred_color == "white":
        return pred_type.capitalize()
    else:
        return pred_type


def get_prediction_fen(splits, learn_empty, learn_color, learn_type):
    fen = ""
    preds = []
    trn_tfms, val_tfms = tfms_from_model(arch, sz)

    for split, i in zip(splits, range(64)):
        if i is not 0 and i % 8 is 0:
            preds.append('/')

        split_im = val_tfms(split)
        preds_empty = learn_empty.predict_array(split_im[None])
        pred_empty = learn_empty.data.classes[np.argmax(preds_empty)]
        if pred_empty != "empty":
            preds_color = learn_color.predict_array(split_im[None])
            pred_color = learn_color.data.classes[np.argmax(preds_color)]
            preds_type = learn_type.predict_array(split_im[None])
            pred_type = learn_type.data.classes[np.argmax(preds_type)]

            pred = get_fen_char(pred_type, pred_color)
            preds.append(pred)
        else:
            if type(preds[-1]) is int:
                preds[-1] += 1
            else:
                preds.append(1)

    for pred in preds:
        fen += str(pred)

    return fen


# A line is given by rho and theta. Given a list of lines, returns a list of
# horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
def hor_vert_lines(lines):
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v


# Given lists of horizontal and vertical lines in (rho, theta) form, returns list
# of (x, y) intersection points.
def intersections(h, v):
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)


# Given a list of points, returns a list of cluster centers.
def cluster(points, max_dist=50):
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = list(map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters))
    return clusters


# Returns closest point to loc.
def closest_point(points, loc):
    dists = np.array(list(map(partial(spatial.distance.euclidean, loc), points)))
    return points[dists.argmin()]


# Given a list of points, returns a list containing the four corner points.
# Tolerance is the amount of error we allow in the uniformity of the points on the grid
# (bigger tolerance = more tolerance)
def find_corners(points, img_dim, tolerance=0.4):
    grid_dist = checker_side_len(img_dim, points)
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    for img_corner in img_corners:
        while True:
            candidate_board_corner = closest_point(points, img_corner)
            points.remove(candidate_board_corner)
            candidate_board_corner_adjacent = closest_point(points, candidate_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(candidate_board_corner),
                                                          np.array(candidate_board_corner_adjacent))
            if (1 - tolerance) * grid_dist < corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(candidate_board_corner)
                board_corners.append(candidate_board_corner)
                break
    return board_corners


# Returns the side-length of the checkered squares on the grid
def checker_side_len(img_dim, points):
    center = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center)
    center_adjacent = closest_point(points, center)
    points.append(center)
    return spatial.distance.euclidean(np.array(center), np.array(center_adjacent))


# Given a list of points, returns the centroid.
def find_centroid(points):
    N = 0
    sumX = 0
    sumY = 0
    for point in points:
        sumX += point[0]
        sumY += point[1]
        N += 1

    # return closest_point(points, (int(sumX / N), int(sumY / N)))
    return [int(sumX / N), int(sumY / N)]


def four_point_transform(img, points, square_length=sz):
    board_length = square_length * 8
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, board_length], [board_length, board_length], [board_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (board_length, board_length))


# Given a board image, returns an array of 64 sub-images.
def split_board_contrast(img):
    contrast_splits = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            norm = img[i * sq_len: (i + 1) * sq_len, j * sq_len: (j + 1) * sq_len]
            cv2.imwrite('temp/norm.jpg', norm)

            split = cv2.imread('temp/norm.jpg')
            clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
            lab = cv2.cvtColor(split, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)  # split on 3 different channels
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            lab = cv2.merge((l2, a, b))  # merge channels
            contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
            contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
            cv2.imwrite('temp/contrast.jpg', contrast)
            contrast_splits.append(open_image('temp/contrast.jpg'))
    return contrast_splits


@app.route('/digitize', methods=['GET'])
def digitize():
    image_bytes = request.files.get("img")
    image_bytes.save("temp/input.jpg")

    # load and scale input image
    image = cv2.imread("temp/input.jpg")
    ratio = image.shape[0] / 1000.0
    orig = image.copy()
    image = imutils.resize(image, height=1000)

    # canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 25, 500)

    # hough line transform
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 100)
    lines = np.reshape(lines, (-1, 2))
    hlines, vlines = hor_vert_lines(lines)

    # intersection detection
    points = intersections(hlines, vlines)
    points = cluster(points, 10)

    # corner detection
    img_shape = np.shape(image)
    points = find_corners(points, (img_shape[1], img_shape[0]))

    # four point perspective transform
    for i in range(len(points)):
        points[i] = (points[i][0] * ratio, points[i][1] * ratio)
    transformed = four_point_transform(orig, points)

    # sub-image split
    contrasts = split_board_contrast(transformed)

    # get fen code prediction from splits
    fen = get_prediction_fen(contrasts, empty_classifier, color_classifier, type_classifier)

    return fen


@app.route('/nextMove', methods=['GET'])
def next_move():
    fen = request.args.get('fen')

    # get best move using stockfish
    stockfish.set_fen_position(fen)
    move, score = stockfish.get_best_move()

    return "{} {} {}".format(move, score.cp, score.mate)


if __name__ == '__main__':
    empty_classifier, color_classifier, type_classifier = init_models()
    stockfish = init_stockfish()
    app.run("0.0.0.0", port=80, threaded=True)
