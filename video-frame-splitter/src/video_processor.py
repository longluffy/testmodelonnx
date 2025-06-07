class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    def split_frames(self, output_folder):
        import cv2
        import os

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video_capture = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        video_capture.release()
        print(f"Extracted {frame_count} frames to {output_folder}")