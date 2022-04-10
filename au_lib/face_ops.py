from imutils import face_utils
import dlib
import cv2

shapePredictorPath = '../../../Dataset/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)
faceDet = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

def get_face_size(image):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return -1, -1
    face = faces[0]
    pos_start = tuple([face.left(), face.top()])
    pos_end = tuple([face.right(), face.bottom()])
    height = (face.bottom() - face.top())
    width = (face.right() - face.left())
    return height, width

def get_facelandmark(image):
    global faceDetector, facialLandmarkPredictor

    face = faceDetector(image, 1)
    if len(face) == 0:
        return None

    shape = facialLandmarkPredictor(image, face[0])
    facialLandmarks = face_utils.shape_to_np(shape)

    xyList = []
    for (x, y) in facialLandmarks[0:]:  # facialLandmarks[17:] without face contour
        xyList.append(x)
        xyList.append(y)
        
    return xyList

def find_faces(image, normalize=False, resize=None, gray=None):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return None

    cutted_faces = [image[face.top():face.bottom(), face.left():face.right()] for face in faces]
    faces_coordinates = [(face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()) for face in faces]

    if normalize:
        if resize is None or gray is None:
            print("Error: resize & gray must be given while normalize is True.")
        normalized_faces = [_normalize_face(face, resize, gray) for face in cutted_faces]
    else:
        normalized_faces = cutted_faces
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face, resize=350, gray=True):
    if gray:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (resize, resize))
    return face