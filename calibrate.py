import numpy as np
import cv2

mouse_pts = []

def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the selected frame
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append([x, y])

def main(image):
    global mouse_pts

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", get_mouse_points)
    cv2.imshow("calibrate", image)

    while(True):
        cv2.waitKey(1)
        if len(mouse_pts) == 4:
            cv2.destroyWindow("calibrate")
            break

    mouse_pts = np.array(mouse_pts, np.int32)
    print("The points are: ", mouse_pts)

    f = open("ref.txt", "w")
    for pt in mouse_pts:
        f.write(str(pt[0])+","+str(pt[1]))
        f.write("\n")
    f.close()

    image = cv2.polylines(image, [mouse_pts.reshape((-1, 1, 2))], True, (255,0,0), 8)
    cv2.imshow("final image", image)
    cv2.waitKey(0)
    cv2.destroyWindow("final image")

    return None


