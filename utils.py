import argparse
import cv2 as cv
import face_recognition
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle
import re
import toolz
import urllib

from collections import Counter
from collections import namedtuple
from mlhub import utils as mlutils

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

MARK_COLOR = (0, 255, 0)  # Green
MARK_WIDTH = 4
TEXT_COLOR = MARK_COLOR
TEXT_WIDTH = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.75

FACE_MODEL = 'hog'
FaceParams = namedtuple('FaceParams', 'scaleFactor minNeighbors minSize')
FACEPARAMS = FaceParams(1.2, 5, 30)  # Default face detection parameters

# ----------------------------------------------------------------------
# File, folder, and I/O
# ----------------------------------------------------------------------

def get_abspath(path):
    """Return the absolute path of <path>.

    Because the working directory of MLHUB model is ~/.mlhub/<model>,
    when user run 'ml score facedetect <image-path>', the <image-path> may be a
    path relative to the path where 'ml score facedetect' is typed, to cope with
    this scenario, mlhub provides mlhub.utils.get_cmd_cwd() to obtain this path.
    """

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        CMD_CWD = mlutils.get_cmd_cwd()
        path = os.path.join(CMD_CWD, path)

    return os.path.abspath(path)


def list_files(path, depth=0):
    """List all files in <path> at level <depth>."""

    path = os.path.join(path, '')
    start = len(path)
    for (root, dirs, files) in os.walk(path):
        segs = root[start:].split(os.path.sep)
        length = len(segs)
        if (depth == 0 and segs[0] == '') or (depth > 0 and length == depth):
            for file in files:
                yield os.path.join(root, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def save_data(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


# ----------------------------------------------------------------------
# Image
# ----------------------------------------------------------------------

def read_cv_image_from(url):
    """Read an image from url or file as grayscale opencv image."""

    return toolz.pipe(
        url,
        urllib.request.urlopen if is_url(url) else lambda x: open(x, 'rb'),
        lambda x: x.read(),
        bytearray,
        lambda x: np.asarray(x, dtype="uint8"),
        lambda x: cv.imdecode(x, cv.IMREAD_COLOR))


def convert_cv2matplot(*images):
    """Convert color space between OpenCV and Matplotlib.

    Because OpenCV and Matplotlib use different color spaces.
    """

    if len(images) > 0:
        res = []
        for image in images:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            res.append(image)

        return res[0] if len(res) == 1 else tuple(res)
    else:
        return None


def _plot_image(ax, img, cmap=None, label=''):
    """Plot <img> in <ax>."""

    ax.imshow(img, cmap)
    ax.tick_params(
        axis='both',
        which='both',
        # bottom='off',  # 'off', 'on' is deprecated in matplotlib > 2.2
        bottom=False,
        # top='off',
        top=False,
        # left='off',
        left=False,
        # right='off',
        right=False,
        # labelleft='off',
        labelleft=False,
        # labelbottom='off')
        labelbottom=False)
    ax.set_xlabel(label)


def plot_side_by_side_comparison(
        leftimg,
        rightimg,
        leftlabel='Original Image',
        rightlabel='Result',
        leftcmap=None,
        rightcmap=None):
    """Plot two images side by side."""

    # Setup canvas

    gs = gridspec.GridSpec(6, 13)
    gs.update(hspace=0.1, wspace=0.001)
    fig = plt.figure(figsize=(7, 3))

    # Plot Left image

    ax = fig.add_subplot(gs[:, 0:6])
    _plot_image(ax, leftimg, cmap=leftcmap, label=leftlabel)

    # Plot right image

    ax = fig.add_subplot(gs[:, 7:13])
    _plot_image(ax, rightimg, cmap=rightcmap, label=rightlabel)

    # Show all of them

    plt.show()


# ----------------------------------------------------------------------
# Command line argument parser
# ----------------------------------------------------------------------

option_parser = argparse.ArgumentParser(add_help=False)
option_parser.add_argument(
    '--scaleFactor',
    type=float,
    default=FACEPARAMS.scaleFactor,
    help='scale factor ({} by default, must > 1)'.format(FACEPARAMS.scaleFactor))
option_parser.add_argument(
    '--minNeighbors',
    type=int,
    default=FACEPARAMS.minNeighbors,
    help='minimum neighbors ({} by default, integer, must > 1)'.format(FACEPARAMS.minNeighbors))
option_parser.add_argument(
    '--minSize',
    type=int,
    default=FACEPARAMS.minSize,
    help='minimum size ({} by default, integer, must > 1)'.format(FACEPARAMS.minSize))


# ----------------------------------------------------------------------
# Face recognition
# ----------------------------------------------------------------------

def detect_faces(rgb, model=FACE_MODEL):
    """Detect all faces in <rgb> using the <model>.

    Args:
        rgb: An RGB image for face detetion.  Note images read by
             OpenCV are type of BGR.
        model (str): The model name to be used for face detection.

    Returns:
        A list of faces found.

    """

    return face_recognition.face_locations(rgb, model=model)


def encode_faces(faces, rgb):
    """Obtain the encodings/embeddings/characteristics of all <faces>
found in the <rgb> image.

    Args:
        faces: a list of face coordinates.
        rgb: An RGB image where <faces> are found.

    Returns:
        A list of encodings of <faces> in <rgb>.
    """

    return face_recognition.face_encodings(rgb, faces)


def _match_face(encoding, candidate_encodings):
    """Compare specific face <encoding> with <candidate_encodings> to find
matches.

    Args:
        candidate_encodings (list): List of known candidate face encodings.
        encoding: a specific face encoding to be recoginised.

    Returns: 
        A list of bool values indicate if <encoding> matches against
        each of candidate_encodings.
    """

    return face_recognition.compare_faces(candidate_encodings, encoding)


def recognise_face(encoding, candidate_encodings, candidate_names):
    """Recognise specific face <encoding> compared with <candidate_data>.

    Args:
        encoding: a specific face encoding to be recoginised.
        candidate_encodings (list): List of known candidate face encodings.
        candidate_names (list): List of known candidate face names.

    Returns:
        The best matched name of <encoding>.
    """

    matches = _match_face(encoding, candidate_encodings)

    if True in matches:
        names = [name for (name, match) in zip(candidate_names, matches) if match]
        cnt = Counter(names)
        return max(cnt, key=cnt.get)
    else:
        return None


def mark_face(image, face, text):
    """Mark the <faces> in <image>.

    Args:
        image: An OpenCV BGR image.
        face: A face cooridinate tuple (top, right, bottom, left)
    """

    # Draw a rectangle around the faces

    top, right, bottom, left = face
    cv.rectangle(image, (left, top), (right, bottom), MARK_COLOR, MARK_WIDTH)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(image, text, (left, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_WIDTH)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

def is_url(url):
    """Check if url is a valid URL."""

    urlregex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(urlregex, url) is not None:
        return True
    else:
        return False
