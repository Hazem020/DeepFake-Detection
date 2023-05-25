import numpy as np
import cv2
from imutils.video import FileVideoStream
from facenet_pytorch import MTCNN
from tensorflow import keras

# create a wrapper class for FastMTCNN


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(
                    f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        return faces

# function to return the lael of the prediction


def get_label(prediction):
    if prediction == 0:
        return "Real"
    else:
        return "Fake"


# function to preprocess each frame of the video


def process_video(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame, (128, 128)) / 255.0


# function to run detection
def run_detection(fast_mtcnn, filename):
    frames = []
    batch_size = 120
    v_cap = FileVideoStream(filename).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

# check if video contain face or not
    face_detected = False
    for j in range(min(batch_size, v_len)):
        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = fast_mtcnn.mtcnn.detect(frame)
        if boxes is not None:
            face_detected = True
            break
    if not face_detected:
        print("Invalid video: Video must contain faces")
        return None

    # if faces detected, continue with deepfake detection
    for j in range(min(batch_size, v_len)):
        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= batch_size or j == v_len - 1:
            faces = fast_mtcnn(frames)
            v_cap.stop()
            return np.expand_dims(np.array(list(map(process_video, faces[:60]))), axis=0)


# create object of FastMTCNN class
fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=50,
    factor=0.6,
    min_face_size=60,
    keep_all=True,
    device='cpu'
)

# load the trained model
model = keras.models.load_model("./vgg.h5")


def start(video_path):
    try:
        # get video array after detection and preprocessing
        cropped_video = run_detection(
            fast_mtcnn, video_path)
        # check if any faces were detected
        if cropped_video is None or len(cropped_video) == 0:
            return {"prediction_label": 'without face'}
        # predict the class of the video
        else:

            prediction = model.predict(cropped_video)
            print(prediction)
            return {
                "prediction_label": get_label(np.argmax(prediction, axis=1))}

    except Exception as e:
        return {"error": str(e)}
