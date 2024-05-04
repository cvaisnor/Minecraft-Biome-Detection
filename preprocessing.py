'''This script is used to turn the videos in /data into frames
    
Original Video Structure
- /data
    - /category1
        - category1_v0.mkv
        ....
        - category1_v1.mkv
    - /cateogry2
        - category2_v0.mkv
        ....
    - ....

Frame Storage Structure
- /frames
    - train
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
            - frame0.jpg
            ....
            - frameN.jpg
        - ....
    - test
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
            - frame0.jpg
            ....
            - frameN.jpg
        - ....
'''

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

# display the number of videos in each category
def basic_vid_stats(dataset_dir):
    for cat in os.listdir(dataset_dir):
        cat_vids = os.path.join(dataset_dir, cat)
        print('The {} class contains {} videos.'.format(cat, len(os.listdir(cat_vids))))
    print()


def trim_video(video_path, seconds):
    '''Funtion trims each end of the video by the number of seconds specified
    Returns: frames of the trimmed video
    '''
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Calculate the number of frames to trim
    trim_frames = int(seconds * fps)
    # Set the start and end points
    start = trim_frames
    end = total_frames - trim_frames
    # Set the start point
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    # Read the video frame by frame and write it to the output video
    frames = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

# given sampled array of frames, physically save frame as JPG images
def store_frames(frames, store_path):
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # convert frame tensor back to BGR format for saving
        path_to_frame = os.path.join(store_path, "frame{}.jpg".format(idx))
        cv2.imwrite(path_to_frame, frame)

# uniform sampling frames for each video
def get_uniform_frames(frames, num_frames_to_sample=1):
    num_frames = len(frames)
    
    uniform_samplied_frames = []

    if num_frames_to_sample > num_frames:
        num_frames_to_sample = num_frames
    
    # uniform random sampling of frames
    # for idx in np.linspace(0, num_frames-1, num_frames_to_sample, dtype=np.int16):
    #     uniform_samplied_frames.append(frames[idx])
    # uniform sampling of frames with progress bar
    for idx in tqdm(np.linspace(0, num_frames-1, num_frames_to_sample, dtype=np.int16)):
        uniform_samplied_frames.append(frames[idx])
    
    # Warning if number of frames is less than required
    if len(uniform_samplied_frames) < num_frames_to_sample:
        print('Warning: Number of frames extracted from video is less than required... no longer uniform sampling')
        # randomly sample frames from the video until n_frames is reached
        while len(frames) < num_frames_to_sample:
            uniform_samplied_frames.append(frames[np.random.randint(0, num_frames-1)])
        
    # final check for equal number of frames
    if len(uniform_samplied_frames) != num_frames_to_sample:
        print('Warning: Number of frames extracted from video is not equal to required number of frames')

    # shuffle the frames
    np.random.shuffle(uniform_samplied_frames)

    return uniform_samplied_frames, num_frames


def create_frame_dataset(dataset_dir, frame_dir, ext='.mkv', n_frames=1, test_size=0.2):
    # create frame storage directory
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    # create train and test directories
    train_dir = os.path.join(frame_dir, 'train')
    test_dir = os.path.join(frame_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # loop through each category in the dataset
    for cat in os.listdir(dataset_dir):
        cat_dir = os.path.join(dataset_dir, cat)
        
        # create train and test directories for each category
        train_cat_dir = os.path.join(train_dir, cat)
        test_cat_dir = os.path.join(test_dir, cat)
        if not os.path.exists(train_cat_dir):
            os.makedirs(train_cat_dir)
        if not os.path.exists(test_cat_dir):
            os.makedirs(test_cat_dir)
        
        # loop through each video in the category
        for vid in os.listdir(cat_dir):
            print('Processing video: {}'.format(vid))
            vid_path = os.path.join(cat_dir, vid)
            frames = trim_video(vid_path, 3) # trim 3 seconds from each end of the video
            uniform_frames, _ = get_uniform_frames(frames, n_frames)

            # split the uniform frames into train and test sets and store them in their respective directories
            split_idx = int(len(uniform_frames) * (1 - test_size))
            train_frames = uniform_frames[:split_idx]
            test_frames = uniform_frames[split_idx:]

            # store frames in train and test directories
            store_frames(train_frames, train_cat_dir)
            store_frames(test_frames, test_cat_dir)
    
    print('Frame extraction and storage complete!')


def main():
    parser = argparse.ArgumentParser(description='Video Dataset Preprocessing')
    parser.add_argument('-dd', '--dataset_dir', help='path of original video dataset', required=True)
    parser.add_argument('-fd', '--frame_dir', help='path to store frames extracted from original video dataset', required=True)
    parser.add_argument('-nf', '--n_frames', type=int, default=1, help='number of frames to be extracted from each video', required=True)
    args = parser.parse_args()
    
    # display basic dataset facts  
    basic_vid_stats(args.dataset_dir)
    
    # random sample and store frames for each video
    create_frame_dataset(args.dataset_dir, args.frame_dir, n_frames=args.n_frames)
    
if __name__=="__main__":
    main()