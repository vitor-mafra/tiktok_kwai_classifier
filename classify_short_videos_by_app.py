import cv2
import imagehash
from PIL import Image
import numpy as np
import os


def is_a_vertical_frame(video_frame):
    width, height = video_frame.size

    if width < height:
        return True
    else:
        return False


def get_last_vertical_frame(video_filename):
    video_capture = cv2.VideoCapture(video_filename)

    last_frame_number = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    last_frame_number = int(last_frame_number) - 1 # frame count starts with 0

    last_vertical_frame = None
    success, image = video_capture.read()
    frame_count = 0
    while success: # while hasn't reach the end of the video
        if frame_count == 0:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            first_frame_image = Image.fromarray(image)

            # if video is not vertical, it cannot be a TikTok video
            if not is_a_vertical_frame(first_frame_image):
                success = False # so we stop processing it
        
        if frame_count == last_frame_number - 12: # last ~0.5s
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            last_vertical_frame = Image.fromarray(image)

        success, image = video_capture.read()
        frame_count += 1
    
    return last_vertical_frame # PIL object


def change_nonwhite_pixels_to_black(image):
    white_inferior_limit = (150, 150, 150)  # everything below this value will be black
    replacement_color = (0, 0, 0)

    image_data = np.array(image)
    # replace all non-white-like color to black
    image_data[(image_data < white_inferior_limit).all(axis=-1)] = replacement_color

    image_with_blacked_nonwhite_pixels = Image.fromarray(image_data, mode="RGB")

    return image_with_blacked_nonwhite_pixels


def get_p_hash(image):
    return imagehash.phash(image)


def get_average_hash(image):
    return imagehash.average_hash(image)


def is_tiktok(last_frame, TIKTOK_LAST_FRAME_P_HASH):
    last_frame_p_hash = get_p_hash(last_frame)

    if last_frame_p_hash - TIKTOK_LAST_FRAME_P_HASH <= 10:  # hamming distance
        return True
    else:
        return False


def is_kwai(last_frame, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH):
    last_frame_p_hash = get_p_hash(last_frame)

    if (last_frame_p_hash - OLDER_KWAI_LAST_FRAME_P_HASH) <= 10: # hamming distance
        return True
    else:
        # so we need to make it all black before calculating the p hash
        treated_last_frame = change_nonwhite_pixels_to_black(last_frame)
        treated_last_frame_p_hash = get_p_hash(treated_last_frame)

        if (treated_last_frame_p_hash - NEWER_KWAI_LAST_FRAME_P_HASH) <= 10:
            return True
        else:
            return False


def is_helo(last_frame, HELO_LAST_FRAME_P_HASH):
    last_frame_p_hash = get_p_hash(last_frame)

    if (last_frame_p_hash - HELO_LAST_FRAME_P_HASH) <= 10:  # hamming distance
        return True
    else:
        return False


# TIKTOK
tiktok_last_frame = get_last_vertical_frame(video_filename=f"./sample_videos/tiktok_example_2.mp4")
TIKTOK_LAST_FRAME_P_HASH = get_p_hash(tiktok_last_frame)

# KWAI
older_kwai = get_last_vertical_frame(video_filename=f"./sample_videos/kwai_example_4.mp4")
newer_kwai = get_last_vertical_frame(video_filename=f"./sample_videos/kwai_example_1.mp4")
treated_newer_kwai = change_nonwhite_pixels_to_black(newer_kwai)

OLDER_KWAI_LAST_FRAME_P_HASH = get_p_hash(older_kwai)
NEWER_KWAI_LAST_FRAME_P_HASH = get_p_hash(treated_newer_kwai)


# HELO
HELO_LAST_FRAME_P_HASH = None

for filename in os.listdir("sample_videos"):
    f = os.path.join("sample_videos", filename)

    if os.path.isfile(f):
        try:
            last_frame = get_last_vertical_frame(f)
            video_is_tiktok = is_tiktok(last_frame, TIKTOK_LAST_FRAME_P_HASH)
            if video_is_tiktok:
                text = str(f).replace("sample_videos/", "")
                tiktok_txt_file = open("is_tiktok.txt", "a")  # append mode
                tiktok_txt_file.write(f"{text} \n")
                tiktok_txt_file.close()
            else:
                video_is_kwai = is_kwai(last_frame, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH)
                if video_is_kwai:
                    text = str(f).replace("sample_videos/", "")
                    kwai_txt_file = open("is_kwai.txt", "a")  # append mode
                    kwai_txt_file.write(f"{text} \n")
                    kwai_txt_file.close()
        except:
            pass
