


import os
import cv2
import numpy as np
import natsort  # pip install natsort
import re

def sorted_pngs_by_number(folder, pattern):
    fnames = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    def key_A(f):
        m = re.match(r"(\d+)_heatmap\.png$", f)
        return int(m.group(1)) if m else float('inf')
    def key_B(f):
        m = re.match(r"frame_(\d+)\.png$", f)
        return int(m.group(1)) if m else float('inf')
    if pattern == 'A': sorted_f = sorted(fnames, key=key_A)
    else: sorted_f = sorted(fnames, key=key_B)
    return [os.path.join(folder, f) for f in sorted_f]

def make_frames(
    input_video_path: str, folder_A: str,
    folder_B: str, tmp_dir: str = "tmp_frames",
    duration_sec: float = 17.0, num_frames: int = 210):
    os.makedirs(tmp_dir, exist_ok=True)
    # load and sort image lists A/B
    imgs_A = sorted_pngs_by_number(folder_A, 'A')[:num_frames]
    imgs_B = sorted_pngs_by_number(folder_B, 'B')[:num_frames]
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_to_read = int(round(fps * duration_sec))
    # sample frame indices evenly across the first duration_sec
    indices = np.linspace(0, total_to_read - 1, num_frames, dtype=int)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(W, H)
    W = W - 200
    for i, vid_idx in enumerate(indices):
        # grab the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
        if not ret: raise RuntimeError(f"Failed to read frame {vid_idx}")
        # load corresponding A/B image or white placeholder
        if i < len(imgs_A): imgA = cv2.imread(imgs_A[i])
        else: imgA = 255 * np.ones_like(frame)
        if i < len(imgs_B): imgB = cv2.imread(imgs_B[i])
        else: imgB = 255 * np.ones_like(frame)
        # resize to match
        imgA = cv2.resize(imgA[300:-300, 600:-600], (W, H))
        imgB = cv2.resize(imgB, (W, H))
        # stitch and save
        combined = cv2.hconcat([frame, imgA, imgB])
        out_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, combined)
    cap.release()
    print(f"Generated {num_frames} frames in '{tmp_dir}'")


def encode_video(frames_dir: str,
    output_video_path: str, duration_sec: float = 10.0):
    # gather frames in order
    frames = natsort.natsorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith('.png')])
    num = len(frames)
    if num == 0:
        raise RuntimeError("No frames to encode!")
    # read one to get size
    sample = cv2.imread(frames[0])
    H, W, _ = sample.shape
    fps = num / duration_sec  # e.g. 210/20 = 10.5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    for path in frames:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    print(f"Saved video '{output_video_path}' at {fps:.2f} fps")

if __name__ == "__main__":
    import numpy as np
    # Paths
    input_video = "video_material/video/Video.mp4"
    folderA = "video_material/image_batch_diagonal_accelerating_img"
    folderB = "video_material/video"
    output_video = "video_material/output_side_by_side.mp4"
    tmp_dir = "video_material/save_video"
    make_frames(input_video, folderA, folderB, tmp_dir, duration_sec=17, num_frames=210)
    encode_video(tmp_dir, output_video, duration_sec = 10.0)
# 54 - 364 
# 210 


"""
Title Frame

Duration: 5 s

Visual: Full-frame title (“Your Project Name” + logo) on a clean background.

Camera Intro Pan

Duration: 15 s

Action:

Start with a close-up on the radar sensor.

Slowly pan out and sweep the camera around the room.

End at a high vantage point showing the entire room in one wide shot.

Overlay: 2D Top-Down Layout

Duration: 5 s

Visual:

In the lower-left corner, fade-in a semi-transparent white top-view floor plan.

Show color-coded labels for walls vs. furniture.

Live Human Motion

Duration: 10 s

Action:

Back to full-room view.

Show a person walking through the scene.

Radar Ray-Trace Illustration

Duration: 10 s

Visual:

Freeze the room view.

Draw a line from the radar sensor to the moving person, then bounce to a wall, and return to the sensor—animated arrowed path.

Split-Screen: Raw + Processed

Duration: 20 s

Layout: Three panels side-by-side in the lower right:

Left: Live room video (human walking).

Middle: Radar heat-map.

Right: Reflector map (processed)

Action: All three update in sync, showing real-time tracking.

Freeze Frame → Input to System

Duration: 10 s

Visual:

Freeze the last multi-panel frame.

Highlight one frame and zoom into the detected points.

Overlay “Input → Our System → Output” callouts.

Finally reveal the 3D ground-truth alignment mesh matching the detected points.

Wi-Fi Scenario 1: Warehouse

Duration: 8 s

Visual:

Simple graphic of a warehouse floor plan.

Animated human icon walking.

Wi-Fi signal emanating from one corner.

Wi-Fi Scenario 2: Bedroom

Duration: 8 s

Visual:

Top-view bedroom layout.

Human icon moving around bed.

Wi-Fi signal icon on a bedside table.

End Frame / Credits

Duration: 5 s

Visual:

“Thank you” + team acknowledgments + contact info.
"""