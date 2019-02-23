import argparse
import cv2 as cv
import face_recognition
import getpass
import hashlib
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle
import re
import readline  # Don't remove !! For prompt of input() to take effect
import sys
import toolz
import urllib.error
import urllib.parse
import urllib.request
import uuid

from collections import Counter
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

FACE_MODEL = 'hog'  # face detection model
FACE_COUNT = 10     # number of photos for creating the database of the same person

# The number of images to return in Bing image search response.  Maximum value for a page is 150.
# See https://docs.microsoft.com/en-sg/rest/api/cognitiveservices/bing-images-api-v7-reference

BING_IMG_SEARCH_PAGE_COUNT = 150
BING_IMG_SEARCH_ENDPOINT = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'

MD5 = 'digests'
FEATURE = 'encodings'


# ----------------------------------------------------------------------
# File, folder, and I/O
# ----------------------------------------------------------------------

def get_abspath(path):
    """Return the absolute path of <path>.

    Because the working directory of MLHUB model is ~/.mlhub/<model>,
    when user run 'ml score facematch <image-path>', the <image-path> may be a
    path relative to the path where 'ml score facematch' is typed, to cope with
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
    """Load python variable from file."""

    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def save_data(data, path):
    """Save Python variable into file."""

    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_key(path):
    """Load subscription key from file."""

    with open(path, 'r') as file:
        key = file.read()

    return key


def save_key(key, path):
    """Save Python variable into file."""

    with open(path, 'w') as file:
        file.write(key)


def download_img(url, folder, prefix):
    """Download image from <url> into <folder> with name as <prefix>_<md5>."""

    # Download image from <url> into a unique file path with <prefix>

    path = os.path.join(folder, get_unique_name(prefix if prefix is not None else 'temp'))
    urllib.request.urlretrieve(url, path)

    # Append md5 digest to image file path

    digest = get_hexdigest(path)
    new_path = change_name_hash(path, digest)
    os.rename(path, new_path)
    return new_path, digest


def get_hexdigest(path):
    with open(path, 'rb') as file:
        digest = hashlib.md5(file.read()).hexdigest()

    return digest


def get_unique_name(prefix):
    """Return a unique name as prefix_uuid."""

    prefix = '_'.join(prefix.split())
    number = str(uuid.uuid4().hex)
    return prefix + '_' + number


def change_name_hash(name, digest):
    """Change name from prefix_uuid to prefix_digest."""

    name = name.split('_')
    name[-1] = digest
    return '_'.join(name)


def get_name_hash(name):
    return name.split('_')[-1]


def make_name_dir(path, name):
    """Createa dir under <path> for person <name> where <name> may contain spaces."""

    name_dir = '_'.join(name.split())
    name_dir_path = os.path.join(path, name_dir)
    os.makedirs(name_dir_path, exist_ok=True)
    return name_dir_path


def ask_for_input(msg):
    msg += "\n> "
    res = input(msg)
    while res.strip() == '':
        res = input("> ")
    return res


def get_url_path_list(url):
    if is_url(url):
        urls = [url]
    else:
        path = get_abspath(url)
        if not os.path.exists(path):
            print("Path not exists: {}".format(url))
            sys.exit(0)

        if os.path.isdir(path):
            urls = list(list_files(path))
        else:
            urls = [path]

    return urls


def stop(msg, status=0):
    print(msg)
    sys.exit(status)


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


def show_image(url, show=True):
    """Read image from <url> and display."""

    img = read_cv_image_from(url)
    img = convert_cv2matplot(img)
    if show:
        display(img)
    return img


def display(img):
    """Display <img> array."""

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
    default=FACE_MODEL,
    help='face detection model ({} by default)'.format(FACE_MODEL))

option_parser.add_argument(
    '--count',
    type=int,
    default=FACE_COUNT,
    help='number of images for training ({} by default, integer, must > 0)'.format(FACE_COUNT))

option_parser.add_argument(
    '--name',
    type=str,
    help='name of the person to match')

option_parser.add_argument(
    '--data',
    type=str,
    help='directory of photos of the same person for training')

option_parser.add_argument(
    '--term',
    type=str,
    help='search term of photos which will be used for face matching')

option_parser.add_argument(
    '--key',
    type=str,
    help='Bing search API subscription key')

option_parser.add_argument(
    '--match',
    type=str,
    help='path or url of photos on which face matching will perform')

option_parser.add_argument(
    '--batch',
    action='store_true',
    help='no interaction')


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


def flatten_encodings(data):
    """Convert <data> into separate list."""

    encoding_list = []
    name_list = []
    for name, meta in data.items():
        encodings = meta[FEATURE]
        encoding_list += encodings
        for i in range(len(encodings)):
            name_list.append(name)

    cnt_dict = Counter(name_list)

    return encoding_list, name_list, cnt_dict


def match_face(encoding, encoding_list, name_list, cnt_dict):
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


def update_face_database(data, name, digest, encodings):
    """Add face encodings into data."""

    if name in data:
        data[name][FEATURE] += encodings
        data[name][MD5].append(digest)
    else:
        data[name] = {}
        data[name][FEATURE] = encodings
        data[name][MD5] = [digest, ]


def check_digest_exist(data, name, digest):
    return name in data and digest in data[name][MD5]


def analyse_face_features(name, img_dir_path, data, model, encode_file, hasdigestag=True):
    print("\nGenerating the face database for '{}'".format(name))
    img_dir_path = get_abspath(img_dir_path)
    for imagePath in list(list_files(img_dir_path)):

        digest = get_name_hash(imagePath) if hasdigestag else get_hexdigest(imagePath)
        if check_digest_exist(data, name, digest):  # Check if encoding is already in the database.
            continue

        print("\n    Detecting faces in the photo:\n        {}".format(imagePath))
        rgb = convert_cv2matplot(read_cv_image_from(imagePath))
        boxes = detect_faces(rgb, model)
        cnt = len(boxes)
        if cnt != 1:
            print("    There are more than one face found!  This photo can not be used.")
            continue

        print("    Calculating the face encodings ...")
        encodings = encode_faces(boxes, rgb)
        update_face_database(data, name, digest, encodings)

    print("\nSaving database ...")
    save_data(data, encode_file)


def recognise_faces(rgb, data, name):
    candidate_encodings, candidate_names, cnt_dict = flatten_encodings(data)
    print("\n        Detecting faces in the image ...")
    boxes = detect_faces(rgb)
    cnt = len(boxes)
    print("            {} face{} found!".format(cnt, 's' if cnt > 1 else ''))
    print("        Calculating the face encodings ...")
    encodings = encode_faces(boxes, rgb)
    print("        Comparing found faces with known faces ...")
    found = False if name is not None else True
    for (box, encoding) in zip(boxes, encodings):
        found_name = match_face(encoding, candidate_encodings, candidate_names, cnt_dict)
        if found_name is not None:
            if found_name == name:
                found = True
            mark_face(rgb, box, found_name)
            print("            '{}' is found in the image!".format(found_name))

    if not found:
        print("            No '{}' are found in the image!".format(name))

    display(rgb)


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
    """Search <term> using Bing image search API <key>."""

    # Construct Bing image search query

    headers = {"Ocp-Apim-Subscription-Key": key}
    params = urllib.parse.urlencode({
        "q": term,
        "offset": offset,
        "count": BING_IMG_SEARCH_PAGE_COUNT,
    })
    url = BING_IMG_SEARCH_ENDPOINT + '?' + params

    # Interpret Bing image search results

    request = urllib.request.Request(url, headers=headers)
    search_results = json.loads(urllib.request.urlopen(request).read())
    total_img_num = search_results["totalEstimatedMatches"]
    img_urls = []
    for v in search_results["value"]:
        img_urls.append(v["contentUrl"])

    return img_urls, total_img_num


def check_key(default_key_file, key_file=None, search_term='cat'):
    """Check subscription key or ask valid key if possible."""

    key = None
    img_urls = None
    total_img_num = 0
    if key_file is None:

        if not os.path.exists(default_key_file):
            msg = """
    To search photos, Bing image search API is used:

        https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/

    And a Bing search API subscription key is needed.

    A 30-days free trail Azure account can be created at:

        https://azure.microsoft.com/en-us/try/cognitive-services/?api=search-api-v7

    You can try it and obtain the key.
"""
            print(msg)
        else:
            yes = mlutils.yes_or_no("A Bing subscription key is found locally! Would you like to use it", yes=True)
            if yes:  # Load Bing search API key if available
                key_file = default_key_file

    else:
        key_file = get_abspath(key_file)
        if not os.path.exists(key_file):  # key_file is exactly the key, not a file
            key = key_file
            key_file = default_key_file
            save_key(key, key_file)

    if key is None and key_file is not None:
        key = load_key(key_file)

    if key is not None:
        try:
            img_urls, total_img_num = search_images(search_term, key)
            save_key(key, default_key_file)
        except urllib.error.URLError:
            print("Maybe the key is wrong!")
            key = None

    if key is None:
        msg = "Please paste the key below (Your key will be kept in\n'{}'):".format(default_key_file)
        print(msg)
        msg = "> "
        while key is None:
            key = getpass.getpass(msg)
            try:
                img_urls, total_img_num = search_images(search_term, key)
                save_key(key, default_key_file)
            except urllib.error.URLError:
                print("Maybe the key is wrong! Please input again.")
                key = None

    return img_urls, total_img_num


def interact_search_for(name, number, key_file, img_dir_path, img_urls=None):
    """Search photos for <name> interactively."""

    msg = ("\nNow photos of '{0}' will be searched by Bing on the Internet.  Found Photos"
           "\nwill be shown one-by-one, you may need to help choosing {1} photos in which"
           "\n'{0}' is the only person in order to set up a face database of '{0}'."
           "\n\nPlease close each window (Ctrl-w) to proceed.")
    print(msg.format(name, number))

    if img_urls is None:
        img_urls, total_img_num = search_images(name, key_file)

    count = 0
    for url in img_urls:
        try:
            msg = "\n    [{}/{}]  Downloading the photo from\n                 {}"
            print(msg.format(str(count + 1).zfill(2), number, url))
            path, _ = download_img(url, img_dir_path, name)
        except (ConnectionResetError, urllib.error.URLError):
            print("             URL access failed!")
            continue

        show_image(path)
        yes = mlutils.yes_or_no("             Is '{}' the only face in the photo or do you want to use it", name, yes=True)
        if yes:
            print("             The photo is saved as\n                 {}".format(path))
            count += 1
            if count == number:
                break
        else:
            os.remove(path)


def interact_get_match_photos(term, img_download_path, default_key_file, key_file):

    if term is not None:
        urls, _ = check_key(default_key_file, key_file, term)
        img_download_path = os.path.join(img_download_path, '_'.join(term.split()))
        os.makedirs(img_download_path, exist_ok=True)
    else:
        msg  = "\n    Please type in a path or URL of a photo for face recognition."
        msg += "\n    Or type in a search term to ask Bing to find a photo for you."
        msg += "\n    Or Ctrl-c to quit.\n"
        print(msg)
        yes = mlutils.yes_or_no("Would you like to use Bing to search a photo for you", yes=True)
        if yes:
            msg = ("\nPlease type in a search term to ask Bing to find a photo for you:"
                   "\n(For example, Satya and Bill Gates)")
            term = ask_for_input(msg)
            urls, _ = check_key(default_key_file, key_file, term)
            img_download_path = os.path.join(img_download_path, '_'.join(term.split()))
            os.makedirs(img_download_path, exist_ok=True)
        else:
            urls = get_url_path_list(ask_for_input("Then please type in path or URL of the photo:"))

    return urls, term, img_download_path
