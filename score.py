print("Loading the required Python modules ...")

import argparse
import cv2 as cv
import os
import readline  # Don't remove !! For prompt of input() to take effect
import urllib.error

from mlhub import utils as mlutils
from utils import (
    VideoStream,
    ask_for_input,
    check_key,
    detect_faces,
    download_img,
    encode_faces,
    flatten_encodings,
    get_url_path_list,
    get_unique_name,
    interact_capture_photos,
    interact_get_match_photos,
    interact_search_for,
    analyse_face_features,
    is_url,
    load_data,
    make_name_dir,
    match_face,
    mark_face,
    option_parser,
    recognise_faces,
    replace_uuid_with_digest,
    show_image,
    stop,
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

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

cwd = os.getcwd()

IMG_PATH = os.path.join(cwd, 'images/score')
os.makedirs(IMG_PATH, exist_ok=True)

SINGLE_FACE_IMG_PATH = os.path.join(IMG_PATH, 'singleface')
os.makedirs(SINGLE_FACE_IMG_PATH, exist_ok=True)

MATCH_FACE_IMG_PATH = os.path.join(IMG_PATH, 'matchface')
os.makedirs(MATCH_FACE_IMG_PATH, exist_ok=True)

VIDEO_PATH = os.path.join(cwd, 'videos/score')
os.makedirs(VIDEO_PATH, exist_ok=True)

ENCODE_PATH = os.path.join(cwd, 'encodings/score')
os.makedirs(ENCODE_PATH, exist_ok=True)
ENCODE_FILE = os.path.join(ENCODE_PATH, 'encodings.pickle')

KEY_PATH = os.path.join(cwd, 'keys/score')
os.makedirs(KEY_PATH, exist_ok=True)
KEY_FILE = os.path.join(KEY_PATH, 'key.txt')

TEMP_PATH = os.path.join(cwd, 'temp/score')
os.makedirs(TEMP_PATH, exist_ok=True)

data = {}
if os.path.exists(ENCODE_FILE):  # Load known faces data if available
    data = load_data(ENCODE_FILE)

# ----------------------------------------------------------------------
# Determine the person's name to match
# ----------------------------------------------------------------------

use_database = False
name = None
if args.name is not None:  # Use the name provided
    name = args.name
elif args.batch:  # Stop if in batch mode
    stop("No name provided!")
else:  # No name provided
    if not args.capture:
        if data != {}:  # Recognise faces of known persons in database
            use_database = True
            name = None
        else:
            name = ask_for_input("\nPlease give a person's name to recognise (For example, Satya)")
    else:
        name = ask_for_input("\nPlease give the person's name to capture")

# ----------------------------------------------------------------------
# Generate face database or load existing one
# ----------------------------------------------------------------------

if not use_database and name in data and args.data is None:  # Face data is available for the name
    if args.batch:
        use_database = True
    else:  # Ask whether or not to use available face data if face database exists
        msg = ("\nYou have searched '{0}' before! So there are face data for '{0}'."
               "\nWould you like to use the data")
        use_database = mlutils.yes_or_no(msg, name, yes=True)

key_file = None
img_download_path = None

hasdigestag = False
if not use_database:  # Face data needs to be obtained from other source instead of existing database
    if args.data is None:  # Search for photos interactively
        print("\n    To recognise '{0}' in arbitrary photos, sample photos of '{0}'\n    are needed.".format(name))
        img_download_path = make_name_dir(SINGLE_FACE_IMG_PATH, name)
        hasdigestag = True
        if args.capture:
            interact_capture_photos(name, args.count, img_download_path)
        else:
            img_urls, _ = check_key(KEY_FILE, args.key, name)  # Get Bing search subscription API key
            key_file = KEY_FILE
            interact_search_for(name, args.count, KEY_FILE, img_download_path, img_urls=img_urls)
    else:  # Use provided photos
        img_download_path = args.data

    analyse_face_features(name, img_download_path, data, args.model, ENCODE_FILE, hasdigestag)
    msg = "\n    Now the face characteristics are remembered!"
else:  # Use existing face database for face matching
    if name is not None and name not in data:  # Face data is not available
        stop("{}'s face information is not available!".format(name))

    msg = "\n    Now the face database are loaded!"

print(msg)

# ----------------------------------------------------------------------
# Recognise the person in a given photo
# ----------------------------------------------------------------------

img_download_path = TEMP_PATH
term = args.term
if key_file is None:
    key_file = args.key
candidate_encodings, candidate_names, cnt_dict = flatten_encodings(data)

if args.video is None and not args.camera:
    if args.match is not None:  # Use given photos to do face matching
        urls = get_url_path_list(args.match)
    else:  # Search for photos
        if args.batch:
            stop("No photos provided!")

        urls, term, img_download_path = interact_get_match_photos(term, MATCH_FACE_IMG_PATH, KEY_FILE, key_file)

    for url in urls:
        if is_url(url):
            try:
                print("\n        Downloading the photo from\n            {}".format(url))
                path, _ = download_img(url, img_download_path, term)
                print("        into\n            {}".format(path))
            except (ConnectionResetError, urllib.error.URLError):
                continue
        else:
            path = url

        if not args.batch:
            rgb = show_image(path)
            yes = mlutils.yes_or_no("\n    Do you want to use this photo", yes=True)
            if not yes:
                os.remove(path)
                continue
        else:
            rgb = show_image(path, show=False)

        recognise_faces(rgb, name, candidate_encodings, candidate_names, cnt_dict)

        if not args.batch:
            yes = mlutils.yes_or_no("\n    Do you want to continue searching for '{}'", term, yes=True)
            if not yes:
                break

else:
    if args.camera:
        src = 0
        print("\nStart recognising faces in live camera ...")
    else:
        src = args.video
        print("\nStart recognising faces in video {} ...".format(src))

    print("\nPlease type 'q' to stop.")
    video = VideoStream(src).start()
    while video.running():
        frame = video.read()
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxes = detect_faces(rgb)
        encodings = encode_faces(boxes, rgb)
        for (box, encoding) in zip(boxes, encodings):
            found_name = match_face(encoding, candidate_encodings, candidate_names, cnt_dict)
            if found_name is not None:
                mark_face(frame, box, found_name)

        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    video.stop()
