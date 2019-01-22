print("Loading the required Python modules ...")
import glob
import os
import pickle

from utils import (
    convert_cv2matplot,
    detect_faces,
    encode_faces,
    list_files,
    load_data,
    mark_face,
    plot_side_by_side_comparison,
    read_cv_image_from,
    recognise_face,
    save_data,
)


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

cwd = os.getcwd()
IMG_PATH = os.path.join(cwd, 'images/demo')
VIDEO_PATH = os.path.join(cwd, 'videos/demo')
ENCODE_PATH = os.path.join(cwd, 'encodings/demo/encodings.pickle')

# Load known faces meta data

data = load_data(ENCODE_PATH)
candidate_encodings = data['encodings']
candidate_names = data['names']


# ----------------------------------------------------------------------
# Image demo
# ----------------------------------------------------------------------

print("\nDemonstrate face recognition using images found in\n{}\n".format(IMG_PATH))
print("Please close each image (Ctrl-w) to proceed through the demonstration.")
for imagePath in list(list_files(IMG_PATH)):

    print("\nRecognising faces in the image:\n  {}".format(imagePath))
    image = read_cv_image_from(imagePath)
    result = image.copy()
    rgb = convert_cv2matplot(image)
    print("    Detecting faces in the image...")
    boxes = detect_faces(rgb)
    cnt = len(boxes)
    print("        {} face{} found!".format(cnt, 's' if cnt > 1 else ''))
    print("    Calculating the face encodings ...")
    encodings = encode_faces(boxes, rgb)
    print("    Comparing found faces with known faces ...")
    for (box, encoding) in zip(boxes, encodings):
        name = recognise_face(encoding, candidate_encodings, candidate_names)
        if name is not None:
            mark_face(result, box, name)

    image, result = convert_cv2matplot(image, result)

    plot_side_by_side_comparison(image, result, rightlabel="Recognised Faces")


# ----------------------------------------------------------------------
# Video demo
# ----------------------------------------------------------------------

print("\nDemonstrate face recognition in a video in \n{}\n".format(VIDEO_PATH))
