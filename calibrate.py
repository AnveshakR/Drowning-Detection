import numpy as np
import cv2

mouse_pts = [] # global array used to store mouse clicks

def get_mouse_points(event, x, y, flags, param): # callback function
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN: # if event is a left click
        mouseX, mouseY = x, y

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append([x, y])

def main(image): # image is selected frame from calibrator function
    global mouse_pts
    image_copy = np.copy(image)
    font_size = 1
    font_thickness = 1

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", get_mouse_points) # get callbacks on selected frame
    cv2.putText(image, "Left click on 4 points to select detection zone", (0, 470), cv2.FONT_HERSHEY_PLAIN, font_size, (0,0,255), font_thickness)
    cv2.imshow("calibrate", image)

    while(True):
        cv2.waitKey(1)
        if len(mouse_pts) == 4: # if 4 points selected, exit loop
            cv2.destroyWindow("calibrate")
            break

    mouse_pts = np.array(mouse_pts, np.int32)

    f = open("ref.txt", "w") # open reference file and save points
    for pt in mouse_pts:
        f.write(str(pt[0])+","+str(pt[1]))
        f.write("\n")
    f.close()

    image = cv2.polylines(image_copy, [mouse_pts.reshape((-1, 1, 2))], True, (255,0,0), 8) # display final image with selected points
    cv2.imshow("final image", image_copy)
    cv2.waitKey(0)
    cv2.destroyWindow("final image")

    return None


