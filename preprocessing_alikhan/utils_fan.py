import copy

import cv2
import numpy as np
import albumentations as A


def remove_text_box_in_video(vid, box_bakcground_pixel, min_rect_area=2_000):
    assert len(vid.shape) == 4

    img = copy.deepcopy(vid[0])

    binary = np.all(img == box_bakcground_pixel, axis=-1).astype(np.uint8)
    binary = binary * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return vid

    contour = max(contours, key = cv2.contourArea)

    # Extract the coordinates of the red rectangle
    x, y, w, h = cv2.boundingRect(contour)

    if w*h >= min_rect_area:
        # print(min_rect_area, w*h)
        vid[:, y:y+h, x:x+w] = 0

    return vid


def pad_to_square(im):
    target_size = max(im.shape[:2])
    return A.PadIfNeeded(
        min_height=target_size,
        min_width=target_size,
        border_mode=0
    )(image=im)["image"]


def get_fan_region(im, threshold=1, video=None):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if isinstance(video, np.ndarray):
        imgray[:5] = 0

    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    contour = max(contours, key = cv2.contourArea)

    # Create a black image with the same size as the original image
    filled_image = np.zeros_like(im)

    # Draw the green contour on the black image as a mask
    cv2.drawContours(filled_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Extract the coordinates of the red rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the original image based on the coordinates of the red rectangle
    cropped_image = im[y:y+h, x:x+w] # FOR IMAGE
    filled_image = filled_image[y:y+h, x:x+w]

    # Bitwise AND the cropped original image with the mask
    masked_image = cv2.bitwise_and(cropped_image, filled_image) # FOR IMAGE

    if isinstance(video, np.ndarray):
        cropped_video = video.copy()[:, y:y+h, x:x+w]
        for i in range(len(cropped_video)):
            cropped_video[i] = cv2.bitwise_and(cropped_video[i], filled_image)

        return cropped_video

    return masked_image
