from torch.utils.data import DataLoader

from src.trueskate_ai.vision.video_dataset import VideoDataset
from utils.data_loader import data_and_words

# Define your data path
data_path = '/Users/ashernoble/Projects/TrueSkate_AI_Training_Data/Training_Data/Sorted/Extracted_Frames'

# Load the data
data, words_list, words_dict = data_and_words(data_path)

# Initialize the dataset
# Initialize the dataset
dataset = VideoDataset(
    data=data,
    num_frames=10,  # Number of frames to extract per video
    transform_frame=None,  # Optional: Transform to apply on individual frames
    transform_video=None  # Optional: Transform to apply on the entire video
)

batch_size = 32  # Set your batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loop through batches
for video_frames, labels in dataloader:
    # Your training code here
    pass