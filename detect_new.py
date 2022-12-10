"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def run(model: str, source: int, width: int, height: int, num_threads: int, calibrate: bool) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
    model: Name of the TFLite object detection model.
    source: Source link to perform detection on.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    calibrate: Set the detection area
    """

    if calibrate:
        calibrator(source, width, height)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
      file_name=model, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
      max_results=10, score_threshold=0.5)
    options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          sys.exit(
              'ERROR: Unable to read from webcam. Please verify your webcam settings.'
          )

        image = cv2.resize(image, (640,480), cv2.INTER_AREA)

        counter += 1

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        for object in detection_result.detections:
            if object.categories[0].category_name == "person":
                b_box = object.bounding_box
                cv2.circle(image, ((b_box.origin_x+b_box.width//2), (b_box.origin_y+b_box.height//2)), 5, (255,0,0), -1)
                #print(object.bounding_box.origin_x)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

    cap.release()
    cv2.destroyAllWindows()

def calibrator(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    while(cap.isOpened()):
        success, frame = cap.read()

        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
            cv2.destroyAllWindows()
            break

    cnt = 0

    def click_event(event, x, y, flags, params):

        global cnt
        if cnt == 3:
            cv2.destroyAllWindows()

        if event == cv2.EVENT_LBUTTONDOWN:
            cnt += 1
            print(x, y)

    cv2.imshow("image", frame)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cnt = 0



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--source', help='Source for detection.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--calibrate',
        help='Set detection area.',
        required=False,
        action='store_true')

    args = parser.parse_args()

    run(args.model, args.source, args.frameWidth, args.frameHeight,
        int(args.numThreads), bool(args.calibrate))


if __name__ == '__main__':
    main()
