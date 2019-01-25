print("Loading the required Python modules ...")

import argparse
import getpass
import os
import readline  # For prompt of input() to take effect
import urllib.error

from mlhub import utils as mlutils
from utils import (
    BING_IMG_SEARCH_PAGE_COUNT,
    FaceParams,
    convert_cv2matplot,
    detect_faces,
    display,
    download_img,
    encode_faces,
    flatten_encodings,
    list_files,
    load_data,
    mark_face,
    option_parser,
    read_cv_image_from,
    recognise_face,
    save_data,
    search_images,
    show_image,
)


# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog='score',
    parents=[option_parser],
    description='Recognise known faces in an image.'
)

args = parser.parse_args()

# Wrap face detection parameters.

face_params = FaceParams(
    args.model,
    args.count,
)


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

cwd = os.getcwd()

IMG_PATH = os.path.join(cwd, 'images/score')
os.makedirs(IMG_PATH, exist_ok=True)

VIDEO_PATH = os.path.join(cwd, 'videos/score')
os.makedirs(VIDEO_PATH, exist_ok=True)

ENCODE_PATH = os.path.join(cwd, 'encodings/score')
os.makedirs(ENCODE_PATH, exist_ok=True)

ENCODE_FILE = os.path.join(ENCODE_PATH, 'encodings.pickle')

# Load known faces meta data if available

data = {}
if os.path.exists(ENCODE_FILE):
    data = load_data(ENCODE_FILE)


# ----------------------------------------------------------------------
# Get a person's name to be searched
# ----------------------------------------------------------------------

name = input("\nPlease give a person's name whose photos can be found on the Internet:\n> ")

img_dir = os.path.join(IMG_PATH, '_'.join(name.split()))
os.makedirs(img_dir, exist_ok=True)


# ----------------------------------------------------------------------
# Get Bing search API subscription key
# ----------------------------------------------------------------------

msg = """
To recognise '{0}' in arbitrary photos, sample photos of '{0}' are needed.
To search photos of '{0}', Bing image search API is used:

    https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/

And a Bing search API subscription key is needed.  a 30-days free trail Azure
account can be created at:

    https://azure.microsoft.com/en-us/try/cognitive-services/?api=search-api-v7
"""
print(msg.format(name))

key = None
while key is None:
    key = getpass.getpass(
        "Then please paste the key below:\n(Don't worry! Your key won't be kept after this task!)\n> ")
    try:
        _, total = search_images(name, key)
    except urllib.error.HTTPError:
        key = None


# ----------------------------------------------------------------------
# Search photos of the person
# ----------------------------------------------------------------------

msg = """
Now photos of '{0}' will be searched by Bing on the Internet.  Found Photos
will be shown one-by-one, you may need to help choosing {1} photos in which
'{0}' is the only person in order to set up a face database of him/her.

Please close each window (Ctrl-w) to proceed.
"""
print(msg.format(name, face_params.count))

count = 0
for offset in range(0, total, BING_IMG_SEARCH_PAGE_COUNT):
    if count == face_params.count:
        break

    image_urls, _ = search_images(name, key, offset=offset)
    for url in image_urls:
        try:
            msg = "\n    [{}/{}]  Downloading the photo from\n               {}"
            print(msg.format(str(count + 1).zfill(2), face_params.count, url))
            path = download_img(url, img_dir, name)
        except urllib.error.HTTPError:
            continue

        image = show_image(path)
        yes = mlutils.yes_or_no("             Is '{}' the only person in the photo", name, yes=True)
        if yes:
            print("             The photo is saved as\n               {}".format(path))
            count += 1
            if count == face_params.count:
                break
        else:
            os.remove(path)


# ----------------------------------------------------------------------
# Update face database for the person
# ----------------------------------------------------------------------

print("\n\nGenerating the face database for '{}'".format(name))

for imagePath in list(list_files(img_dir)):
    print("\n    Detecting faces in the photo:\n      {}".format(imagePath))
    image = read_cv_image_from(imagePath)
    result = image.copy()
    rgb = convert_cv2matplot(image)
    boxes = detect_faces(rgb)
    cnt = len(boxes)
    if cnt != 1:
        print("        There are more than one face found!  This photo can not be used.")
        continue

    print("    Calculating the face encodings ...")
    encodings = encode_faces(boxes, rgb)
    if name in data:
        data[name] += encodings
    else:
        data[name] = encodings

print("\nSaving database ...")
save_data(data, ENCODE_FILE)
candidate_encodings, candidate_names, cnt_dict = flatten_encodings(data)


# ----------------------------------------------------------------------
# Recognise the person in a given photo
# ----------------------------------------------------------------------

msg = """
Now the face characteristics are remembered!
'{0}' can be found in arbitrary photos.

Please type in a path or URL of a photo to see if '{0}' is there.
Or type in a search term to ask Bing to find a photo for you.
Or Ctrl-c to quit.
"""
print(msg.format(name))
url = ''
while url == '':
    yes = mlutils.yes_or_no("Do you want to use Bing to search a photo for you", yes=True)
    if not yes:
        url = input("Then please type in path or URL of the photo:\n> ")
    else:
        term = input("Please type in a search term to ask Bing to find a photo for you:\n> ")
        _, total = search_images(term, key)

        img_dir = os.path.join(img_dir, '_'.join(term.split()))
        os.makedirs(img_dir, exist_ok=True)

        for offset in range(0, total, BING_IMG_SEARCH_PAGE_COUNT):
            image_urls, _ = search_images(term, key, offset=offset)
            for url in image_urls:
                try:
                    path = download_img(url, img_dir, term)
                except urllib.error.HTTPError:
                    continue

                rgb = show_image(path)
                yes = mlutils.yes_or_no("\n    Do you want to use this photo", yes=True)
                if not yes:
                    os.remove(path)
                    continue

                print("        Detecting faces in the image ...")
                boxes = detect_faces(rgb)
                cnt = len(boxes)
                print("            {} face{} found!".format(cnt, 's' if cnt > 1 else ''))
                print("        Calculating the face encodings ...")
                encodings = encode_faces(boxes, rgb)
                print("        Comparing found faces with known faces ...")
                for (box, encoding) in zip(boxes, encodings):
                    found_name = recognise_face(encoding, candidate_encodings, candidate_names, cnt_dict)
                    if found_name is not None:
                        mark_face(rgb, box, found_name)
                        display(rgb)
                    else:
                        msg = "            No '{}' or other known persons are found in the image!"
                        print(msg.format(name))
