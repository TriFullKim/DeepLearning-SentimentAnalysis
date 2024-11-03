from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist

from modules.util import save_json, now
from modules.train import (
    clean_text,
    word_to_num,
    seq_padding,
    dataloader_gen,
    SentimentAnalysisRNN,
    calculate_accuracy,
    evaluate,
)


IMDB = pd.read_csv("./database/IMDB Dataset.csv")

# 텍스트 데이터 전처리
IMDB = IMDB.drop_duplicates(subset="review")
IMDB.loc[:, "sentiment"] = IMDB["sentiment"].map({"positive": 1, "negative": 0})
IMDB["review_cleaned"] = IMDB["review"].apply(clean_text)

# Tokenization
tokenizer = TreebankWordTokenizer()
IMDB.loc[:, "review_tokenized"] = IMDB.loc[:, "review_cleaned"].apply(tokenizer.tokenize)

# 단어집합 생성
all_tokens = np.hstack(IMDB["review_tokenized"])

# 단어 집합 생성 및 빈도 계산
vocab = FreqDist(all_tokens)

# Hard Threshold by Frequency
FREQ_THRESHOLD = 3
vocab = {key: value for key, value in vocab.items() if value >= FREQ_THRESHOLD}

# 단어와 인덱스 할당
word2idx = {word: idx + 2 for idx, (word, _) in enumerate(vocab.items())}
word2idx["<pad>"] = 0  # 패딩을 위한 인덱스 0 예약
word2idx["<unk>"] = 1  # 알 수 없는 단어를 위한 인덱스 1 예약
VOCAB_SIZE = len(word2idx)


# 맵핑(단어 집합을 데이터에 적용)
IMDB["review_numbered"] = IMDB["review_tokenized"].apply(
    lambda _X: [word_to_num(word, word2idx) for word in _X]
)
IMDB["token_length"] = IMDB["review_numbered"].apply(lambda _X: len(_X))


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 토큰 최대 길이 설정
MAX_TOKEN = 256
IMDB_256 = IMDB[IMDB["token_length"] < MAX_TOKEN]

# Train/Test Split
x_raw, y_raw = IMDB_256["review_numbered"], IMDB_256["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    x_raw, y_raw, test_size=0.8, random_state=0, stratify=y_raw
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)


# 토큰 패딩
X_train, X_valid, X_test = seq_padding(X_train), seq_padding(X_valid), seq_padding(X_test)
y_train, y_valid, y_test = y_train.apply(int), y_valid.apply(int), y_test.apply(int)


# 배치 학습을 위한 토치 데이터 셋 정의
BATCH_SIZE = 256
dataloader = dataloader_gen(X_train, y_train.to_numpy(), BATCH_SIZE)
valid_dataloader = dataloader_gen(X_valid, y_valid.to_numpy(), BATCH_SIZE)
test_dataloader = dataloader_gen(X_test, y_test.to_numpy(), BATCH_SIZE)


# Parameter Settings
EMBED_SIZE = 256
HIDDEN_SIZE = 1024
OUTPUT_DIM = 2


# Training setup
N_EPOCHS = 300
LEARNING_RATE = 0.00001
model = SentimentAnalysisRNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_DIM, device=device).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Model
# https://wikidocs.net/217083 : Evaluation Function
trainin_at = now()
best_val_loss = float("inf")
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()

    for batch_idx, samples in enumerate(dataloader):
        x_batch, y_batch = samples
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        predicted = model(x_batch)
        loss = loss_fn(predicted, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += calculate_accuracy(predicted, y_batch) * y_batch.size(0)
        train_total += y_batch.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(dataloader)

    # Validation
    val_loss, val_accuracy = evaluate(model, valid_dataloader, loss_fn, device)

    print(f"Epoch {epoch+1}/{N_EPOCHS}:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 검증 손실이 최소일 때 체크포인트 저장
    if val_loss < best_val_loss:
        print(
            f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다."
        )
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model_checkpoint-{trainin_at}.pth")


save_json(
    f"./training_param_{trainin_at}.json",
    {
        "voca_frequency_thresold": FREQ_THRESHOLD,
        "token_truncation": MAX_TOKEN,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "embed_dim": EMBED_SIZE,
        "rnn_hidden_dim": HIDDEN_SIZE,
        "output_dim": OUTPUT_DIM,
        "loss": best_val_loss,
    },
)
