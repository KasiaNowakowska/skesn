"""
python script to generate avi

Usage: makeavi.py <input_path> <output_gif> [--fps=<fps>]

Options:
    --fps=<fps>    frames per second [default: 1]

"""

from PIL import Image
import cv2
import os
from docopt import docopt
args = docopt(__doc__)


def create_avi(input_path, output_avi, fps):
    # List all files in the input image folder
    files = sorted(os.listdir(input_path))
    
    # Get the first image to initialize video writer
    first_image = cv2.imread(os.path.join(input_path, files[0]))
    height, width, _ = first_image.shape
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use desired codec (e.g., XVID)
    out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
    
    # Iterate through each image and write to the video
    for file_name in files:
        if file_name.endswith('.png'):
            file_path = os.path.join(input_path, file_name)
            frame = cv2.imread(file_path)
            out.write(frame)
    
    # Release the VideoWriter object
    out.release()
    print(f"AVI video created: {output_avi}")


# Extract arguments
input_dir = args['<input_path>']
output_gif = args['<output_gif>']
fps = float(args['--fps'])

#if not os.path.exists(output_path):
 #   os.makedirs(output_path)
  #  print('made directory')
    
# Create GIF from images in the specified directory
create_avi(input_dir, output_gif, fps)
