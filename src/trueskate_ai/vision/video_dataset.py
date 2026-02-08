import os
import sys

import numpy as np
import torch
from PIL import Image
from torch._C._instruction_counter import end
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class VideoDataset(Dataset):
    def __init__(self, data, num_frames=10, max_frames=500, transform_frame=None, transform_video=None, output_video_path=None):
        self.data = data  # It should be a list of tuples where data[0] is the path to the video and data[1] is the label

        self.transform_frame = transform_frame  # Transformations to be done on the individual frames.
        # Recommended to use when transforms is required at frames level with some randomness, eg: Random Crop
        # Note: If the Dataset is showing tensor issue, try adding `ToTensor()` in transform.

        self.transform_video = transform_video  # Transformations to be done on the whole video.
        # Recommended to use when transforms is required at video level with some randomness, eg: Random Horizontal Flip

        self.num_frames = num_frames  # Number of frames to be extracted from the video

        self.max_frames = max_frames

        self.output_video_path = output_video_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        video_path = self.data[idx][0]

        video_frames = []

        video_images = sorted(os.listdir(video_path))

        frame_positions = np.linspace(0, end, self.max_frames, endpoint=False)

        # Selecting the NUM_FRAMES from the video
        for n in frame_positions:
            img = video_images[int(n)]
            img_path = os.path.join(self.output_video_path, img)
            with Image.open(img_path) as pil_img:
                if self.transform_frame:
                    video_frames.append(self.transform_frame(
                        pil_img))  # Note: Try adding ToTensor() in transform_frame, if any tensor related error arrises.

                else:
                    video_frames.append(to_tensor(pil_img))

        try:
            video_frames = torch.stack(
                video_frames)  # Note: Try adding ToTensor() in transform_frame, if any tensor related error arrises.
        except TypeError:
            print(f"TypeError: Tried to stack {type(video_frames[0])}. Add ToTensor() in transform_frame!")
            sys.exit(1)
            return None, None

        if self.transform_video:
            video_frames = self.transform_video(video_frames)

        label = self.data[idx][1]
        return video_frames, label