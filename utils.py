import argparse
import cv2 as cv
import face_recognition
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle
import re
import toolz
import urllib.parse
import urllib.request
import uuid

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

FaceParams = namedtuple('FaceParams', 'model count')
FACEPARAMS = FaceParams('hog', 10)  # Default face detection parameters

BING_IMG_SEARCH_PAGE_COUNT = 50
BING_IMG_SEARCH_ENDPOINT = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'


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
        path = os.path.join(mlutils.get_cmd_cwd(), path)

    return os.path.abspath(path)


def list_files(path, depth=0):
    """List all files in <path> at level <depth>.  If depth < 0, list all files under <path>."""

    path = os.path.join(path, '')
    start = len(path)
    for (root, dirs, files) in os.walk(path):
        if depth < 0:
            for file in files:
                yield os.path.join(root, file)
        else:
            segs = root[start:].split(os.path.sep)
            if (depth == 0 and segs[0] == '') or len(segs) == depth:
                for file in files:
                    yield os.path.join(root, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def save_data(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def download_img(url, folder, prefix):
    path = os.path.join(folder, get_unique_name(prefix))
    urllib.request.urlretrieve(url, path)
    return path


def get_unique_name(prefix):
    prefix = '_'.join(prefix.split())
    number = str(uuid.uuid4().hex)
    return prefix + '_' + number


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


def show_image(url):
    img = read_cv_image_from(url)
    img = convert_cv2matplot(img)
    display(img)
    return img


def display(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()


# ----------------------------------------------------------------------
# Command line argument parser
# ----------------------------------------------------------------------

option_parser = argparse.ArgumentParser(add_help=False)
option_parser.add_argument(
    '--model',
    type=str,
    default=FACEPARAMS.model,
    help='face recognition model ({} by default)'.format(FACEPARAMS.model))
option_parser.add_argument(
    '--count',
    type=int,
    default=FACEPARAMS.count,
    help='number of images for training ({} by default, integer, must > 0)'.format(FACEPARAMS.count))


# ----------------------------------------------------------------------
# Face recognition
# ----------------------------------------------------------------------

def detect_faces(rgb, model=FACEPARAMS.model):
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


def flatten_encodings(data):
    encoding_list = []
    name_list = []
    for name, encodings in data.items():
        encoding_list += encodings
        for i in range(len(encodings)):
            name_list.append(name)

    cnt_dict = Counter(name_list)

    return encoding_list, name_list, cnt_dict


def recognise_face(encoding, encoding_list, name_list, cnt_dict):
    """Recognise specific face <encoding> compared with <candidate_data>.

    Args:
        encoding: a specific face encoding to be recoginised.
        encoding_list: List of known candidate face encodings.
        name_list (list): List of known candidate face names.
        cnt_dict (dict): Number of encodings for each name in <name_list>.

    Returns:
        The best matched name of <encoding>.
    """

    matches = _match_face(encoding, encoding_list)

    if True in matches:
        names = [name for (name, match) in zip(name_list, matches) if match]
        cnt = Counter(names)
        return max(cnt, key=lambda name: cnt[name]/cnt_dict[name])
    else:
        return None


def mark_face(image, face, text):
    """Mark the <faces> in <image>.

    Args:
        image: An OpenCV BGR image.
        face: A face cooridinate tuple (top, right, bottom, left).
        text: Text would be displayed above the face box.
    """

    # Draw a rectangle around the faces

    top, right, bottom, left = face
    cv.rectangle(image, (left, top), (right, bottom), MARK_COLOR, MARK_WIDTH)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(image, text, (left, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_WIDTH)


# ----------------------------------------------------------------------
# URL
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


# ----------------------------------------------------------------------
# Bing image search
# ----------------------------------------------------------------------

def search_images(term, key, offset=0):
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = urllib.parse.urlencode({
        "q": term,
        "offset": offset,
        "count": BING_IMG_SEARCH_PAGE_COUNT,
    })
    url = BING_IMG_SEARCH_ENDPOINT + '?' + params
    request = urllib.request.Request(url, headers=headers)
    search_results = json.loads(urllib.request.urlopen(request).read())
    total_img_num = search_results["totalEstimatedMatches"]
    img_urls = []
    for v in search_results["value"]:
        img_urls.append(v["contentUrl"])

    return img_urls, total_img_num
