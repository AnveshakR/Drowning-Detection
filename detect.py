"""Main script to run the object detection routine."""
import argparse
import sys
import time
import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import calibrate as calibration

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

    try:
        f = open("ref.txt", "r")
    except FileNotFoundError:
        calibration_handler(source, width, height)


    if calibrate:
        calibration_handler(source, width, height)


    f = open("ref.txt", "r")
    detection_zone = [line.rstrip() for line in f]
    detection_zone = [list(map(int, i.strip().split(','))) for i in detection_zone]
    detection_zone = np.array(detection_zone, np.int32)
    print("detection_zone", detection_zone)
    f.close()

    detection_zone = detection_zone.reshape((-1, 1, 2))

    real_count = 0
    expect_count = 0

    # Variables to calculate FPS
    fps_counter, fps = 0, 0
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
        image = cv2.polylines(image, [detection_zone], True, (255, 0, 0), 3)
        fps_counter += 1

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        for object in detection_result.detections:
            if object.categories[0].category_name == "person":
                b_box = object.bounding_box
                center = [(b_box.origin_x+b_box.width//2), (b_box.origin_y+b_box.height//2)]

                position = cv2.pointPolygonTest(detection_zone, center, False)

                if position == 1:
                    real_count += 1
                elif position == 0:
                    real_count -= 1

                cv2.circle(image, center, 5, (255,0,0), -1)
                cv2.putText(image, str(position), (center[0]+5,center[1]), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)


        # Calculate the FPS
        if fps_counter % fps_avg_frame_count == 0:
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

def calibration_handler(source, width, height):
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
            cap.release()
            cv2.destroyAllWindows()
            break

    calibration.main(frame)
    print("hi")

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
