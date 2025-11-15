# src/frame_sampling.py

import cv2
import os

def sample_frames(input_video_path, scenes, num_frames=3, output_dir="./output/frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)

    for scene_idx, scene in enumerate(scenes):
        start, end = scene["start"], scene["end"]
        frame_count = end - start + 1
        step = max(frame_count // num_frames, 1)

        frames_list = []

        for i, f in enumerate(range(start, end + 1, step)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_filename = os.path.join(output_dir, f"scene{scene_idx}_frame{i}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames_list.append(frame_filename)

        scene["frames"] = frames_list

    cap.release()
    return scenes
