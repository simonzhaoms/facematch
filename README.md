# Simple Face Recognition #

This is a simple face recognition example of using deep learning to
recognise faces within a picture.  It originates from Adrian
Rosebrock's article --
[Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/).

See the GitHub repository for examples of its usage:
https://github.com/simonzhaoms/facematch


## Usage ##

* To install and demonstrate the algorithm:

  ```console
  $ pip3 install mlhub
  $ ml install   facematch
  $ ml configure facematch
  $ ml demo      facematch
  ```

* To match you in camera:

  ```console
  $ ml score facematch --capture --camera
  ```

  It will open your camera to capture 5 photos of you to generate your
  face database, then recognise you in a live camera video.
  
* You can also provide the path or URL of a person's photos via option
  `--data`, and let facematch to recognise him/her in a photo via the
  option `--match`:

  ```console
  $ ml score facematch --data <photo-of-the-person> --match <photo-for-recognition>
  ```

   or video via the option `--video`:

  ```console
  $ ml score facematch --data <photo-of-the-person> --video <video-for-recognition>
  ```


## More details ##

### About collecting photos ###

The photos used for recognition here are collected by using
[Bing image search API](https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/).  The code for collecting photos is adapted from
[How to (quickly) build a deep learning image dataset](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/).

In the interactive mode of `ml score facematch`, a subscription key of
Bing image search API is required.  You can get 7-days free account
together with a subscription key at [Try Microsoft Azure Cognitive
Services](https://azure.microsoft.com/en-us/try/cognitive-services/?api=search-api-v7).

More details about how to use Bing image search API can be found at
* [Bing Image Search API Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/)
* [Quickstart: Search for images using the Bing Image Search REST API and Python](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/quickstarts/python)
* [Image Search API v7 reference](https://docs.microsoft.com/en-sg/rest/api/cognitiveservices/bing-images-api-v7-reference)

