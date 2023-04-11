import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"

# Set paths to data and model directories
data_path = "../data/video"
annotation_path = "../label/final.txt"
amount = 30 + 1
# Define a function to extract frames from a video file
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    #num_frames = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if count % 8 == 0:  # Only keep frames at 3fps
                frame = cv2.resize(frame, (224, 224))  # Resize the frame to (224, 224)
                frame = torch.from_numpy(frame)
                frames.append(frame)
                #num_frames += 1
        else:
            break
    cap.release()
    #print(f"Number of frames extracted from {video_path}: {num_frames}")
    print(f"aaaaaaaaaaaaaaaaaaaaaaaaaa{frames}")
    return torch.stack(frames)

# Define a function to preprocess the video frames
def preprocess_frames(frames):
    # Resize the frames to (112, 112) and convert to float32
    frames = torch.nn.functional.interpolate(frames, (112, 112)).numpy().astype('float32')

    # Normalize the pixel values to the range [0, 1]
    frames /= 255.0

    # Move the frames to the specified device
    frames = torch.from_numpy(frames).to(device)

    return frames

# Define a function to generate a sentence based on the annotations and video
def generate_sentence(annotations, video, tokenizer, model):
    # Convert the annotations to a string
    annotations_string = " ".join(annotations)
    print('하나')
    # Convert the video tensor to a string
    video_string = str(video.numpy().tolist())
    print('둘')
    # Concatenate the annotations string and video string
    input_text = annotations_string + video_string
    print('셋')
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    print('네')
    # Move the input_ids tensor to the specified device
    input_ids = input_ids.to(device)
    print('다섯')
    # Move the model to the specified device
    model.to(device)
    print('여섯')
    # Generate a sentence using the model
    output = model.generate(input_ids, max_length=50, do_sample=True)
    print('일곱')
    # Decode the output and return the generated sentence
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print('여덟')
    return output_text

# Define a generator function to load data in batches
def data_generator(data_path, annotation_path, videos, batch_size, max_frames=80):
    with open(annotation_path, 'r') as f:
        x = f.readlines()
        annotations = [line.strip().split(',')[:7] for line in x[:amount]]
        ground_truth = [line.strip().split(',')[7] for line in x[:amount]]
    while True:
        data = []
        labels = []
        for filename in np.random.permutation(videos):
            video_path = os.path.join(data_path, filename)
            frames = extract_frames(video_path)
            if frames.shape[0] > max_frames:
                frames = frames[:max_frames]  # Truncate frames
            else:
                pad_width = (0, max_frames - frames.shape[0], 0, 0, 0, 0)
                frames = torch.nn.functional.pad(frames, pad_width, 'constant')  # Pad frames
            frames = preprocess_frames(frames)  # Preprocess the frames
            data.append(frames)
            idx = videos.index(filename)
            label = ground_truth[idx]  # Load the ground truth label for the video
            labels.append(label)
            annotation = annotations[idx]  # Load the annotation for the video

            print(f"@@@@@@@@@@@@@@@@@@{len(data)}")
            if len(data) == batch_size:
                # Generate a sentence for each batch using the annotations and video
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                print('one')
                model = GPT2LMHeadModel.from_pretrained(model_name)
                print('two')
                batch_sentences = [generate_sentence(annotation, video, tokenizer, model) for annotation, video in
                                   zip(annotations, data)]
                print('three')
                batch_input_ids = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)[
                    'input_ids']
                print('four')
                yield torch.cat([torch.stack(data).to(device), torch.tensor(labels).to(device)], dim=1), torch.tensor(batch_input_ids).to(device)
                print(1)
            print("@@@@@@@@@@@@@")
# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define the fine-tuning parameters
epochs = 2
batch_size = 8
learning_rate = 5e-5

# Load the pre-trained GPT-2 model and configure the output layer
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss = nn.CrossEntropyLoss()

# Split data into training and validation sets
all_videos = os.listdir(data_path)[:amount]
num_train_videos = int(len(all_videos) * 0.8)
train_videos = all_videos[:num_train_videos]
val_videos = all_videos[num_train_videos:]

# Define the training and validation data generators
train_generator = data_generator(data_path, annotation_path, train_videos, batch_size=batch_size)
val_generator = data_generator(data_path, annotation_path, val_videos, batch_size=batch_size)

# Train the model
for epoch in range(epochs):
    train_loss = torch.tensor(0.0, device=device)
    train_steps = torch.tensor(0, device=device)
    val_loss = torch.tensor(0.0, device=device)
    val_steps = torch.tensor(0, device=device)

    # Train on the training set
    model.train()

    for data, labels, batch_sentences in train_generator:
        optimizer.zero_grad()
        input_ids = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)['input_ids']
        input_ids = input_ids.to(device) # move input_ids to the device
        outputs = model(input_ids=input_ids, labels=input_ids)
        batch_loss = loss(outputs.logits.view(-1, model.config.vocab_size), input_ids.view(-1))
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()
        train_steps += 1

    # Evaluate on the validation set
    model.eval()
    with torch.no_grad():
        for data, labels, batch_sentences in val_generator:
            input_ids = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)['input_ids']
            input_ids = input_ids.to(device) # move input_ids to the device
            outputs = model(input_ids=input_ids, labels=input_ids)
            batch_loss = loss(outputs.logits.view(-1, model.config.vocab_size), input_ids.view(-1))

            val_loss += batch_loss.item()
            val_steps += 1

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss / train_steps}, Val Loss = {val_loss / val_steps}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'D:/pythonProject/model_save/final_retrain_model.pth')