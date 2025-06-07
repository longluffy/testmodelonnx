def create_output_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_valid_video_file(file_path):
    import os
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return os.path.isfile(file_path) and file_path.endswith(valid_extensions)