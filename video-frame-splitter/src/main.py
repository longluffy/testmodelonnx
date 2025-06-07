def main():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from video_processor import VideoProcessor
    from config import OUTPUT_DIR

    video_path = input("Enter the path to the video file: ")

    if not os.path.isfile(video_path):
        print("The specified video file does not exist.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    processor = VideoProcessor(video_path)
    processor.split_frames(OUTPUT_DIR)
    print(f"Frames have been successfully saved to the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()