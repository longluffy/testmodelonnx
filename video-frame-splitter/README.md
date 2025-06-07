# Video Frame Splitter

This project is a simple Python application that splits all frames from an input video and saves them to a specified folder. The output frames are stored in a directory named `splitted`.

## Project Structure

```
video-frame-splitter
├── src
│   ├── main.py          # Entry point of the program
│   ├── video_processor.py # Contains the VideoProcessor class for frame splitting
│   └── utils.py        # Utility functions for directory management
├── requirements.txt     # Lists the required dependencies
├── config.py            # Configuration settings
└── README.md            # Project documentation
```

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

The main dependency for this project is `opencv-python`, which is used for video processing.

## Usage

1. Ensure you have Python installed on your machine.
2. Clone the repository or download the project files.
3. Navigate to the project directory.
4. Run the main program:

```
python src/main.py
```

5. Follow the prompts to enter the path of the video file you want to process.

## Output

All frames from the input video will be saved in a folder named `splitted` within the project directory. If the folder does not exist, it will be created automatically.

## License

This project is open-source and available under the MIT License.