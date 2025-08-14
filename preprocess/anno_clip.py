import json
import os.path as osp
from glob import glob

import cv2
import numpy as np
from fire import Fire



def batch_annote(image_dir):
    video_list = glob(osp.join(f"{image_dir}/*.mp4"))
    for video_path in video_list:
        annotate(video_path)

    return


def print_instruction():
    print("""
    Press 's' to annotate start frame
    Press 'e' to annotate end frame
    Press 'q' to save annotations and exit
    """)


def annotate(video_path="sample_video.mp4"):
    print_instruction()
    cap = cv2.VideoCapture(video_path)
    anno_file = video_path.replace(".mp4", ".json")
    start_frames = []
    end_frames = []
    if osp.exists(anno_file):
        anno = json.load(open(anno_file))
        start_frames = anno["start_frames"]
        end_frames = anno["end_frames"]
    print(f"Start frames: {start_frames}, {len(start_frames)}")
    print(f"End frames: {end_frames}, {len(end_frames)}")

    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Total Frames: {total_frames}, FPS: {fps}")

        # Initialize annotation lists

        # Create a simple slider using OpenCV window
        def on_trackbar(val):
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret, frame = cap.read()
            if ret:
                combined_img = combine_images(frame)
                cv2.imshow("Video & Annotations", combined_img)

        def combine_images(frame):
            # Create a blank image for the trackbar with annotations
            trackbar_img = np.ones((50, total_frames, 3), dtype=np.uint8) * 255

            # Draw blue dots for start frames
            for frame_num in start_frames:
                if frame_num < total_frames:
                    cv2.circle(
                        trackbar_img, (frame_num, 25), 5, (255, 0, 0), -1
                    )  # Changed to red for visibility

            # Draw red dots for end frames
            for frame_num in end_frames:
                if frame_num < total_frames:
                    cv2.circle(trackbar_img, (frame_num, 25), 5, (0, 0, 255), -1)

            # Combine the video frame and the annotation track
            # resize to the same width
            trackbar_img = cv2.resize(
                trackbar_img, (frame.shape[1], trackbar_img.shape[0])
            )
            combined_img = np.vstack(
                (
                    trackbar_img,
                    frame,
                )
            )
            return combined_img

        cv2.namedWindow("Video & Annotations")
        cv2.createTrackbar(
            "Time Slider", "Video & Annotations", 0, total_frames - 1, on_trackbar
        )

        # Display the first frame
        ret, frame = cap.read()
        if ret:
            combined_img = combine_images(frame)
            cv2.imshow("Video & Annotations", combined_img)

        # Main loop to handle user input
        while True:
            key = cv2.waitKey(1) & 0xFF

            # Annotate start frame
            if key == ord("s"):
                current_frame = cv2.getTrackbarPos("Time Slider", "Video & Annotations")
                start_frames.append(current_frame)
                print(f"Start frame annotated at frame {current_frame}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if ret:
                    combined_img = combine_images(frame)
                    cv2.imshow("Video & Annotations", combined_img)

            # Annotate end frame
            elif key == ord("e"):
                current_frame = cv2.getTrackbarPos("Time Slider", "Video & Annotations")
                end_frames.append(current_frame)
                print(f"End frame annotated at frame {current_frame}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if ret:
                    combined_img = combine_images(frame)
                    cv2.imshow("Video & Annotations", combined_img)

            # Save annotations and exit
            elif key == ord("q"):
                # Save annotations to files
                with open(anno_file, "w") as f:
                    json.dump(
                        {"start_frames": start_frames, "end_frames": end_frames}, f
                    )

                print("Annotations saved. Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # this is for aria to annotate clip
    Fire(batch_annote)
