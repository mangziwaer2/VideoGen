import os.path

import cv2


def cut_vid2img(vid_path):
    cap=cv2.VideoCapture(vid_path)
    save_path_dir=os.path.join(os.path.dirname(vid_path),"frames")
    os.makedirs(save_path_dir,exist_ok=True)
    frame_length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i=-1
    while True:

        success,img=cap.read()

        if not success:
            break

        i+=1

        save_path=os.path.join(save_path_dir,F"{i:05d}.png")

        cv2.imwrite(save_path,img)


if __name__ == '__main__':

    path="../datasets/vid/1/video.mp4"

    cut_vid2img(path)
