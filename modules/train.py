import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()  # 소문자화
    text = re.sub(r"<.*?>", "", text)  # HTML태그 제거
    text = re.sub(r"[^.a-z\s!?']", "", text)  # 특수 문자 및 숫자 제거
    text = " ".join(word for word in text.split() if word not in stop_words)  # 불용어 제거
    text = re.sub(r'([!?\'"])\1+', r"\1", text)  # !?'이 2개 이상이면 한 개로 만들어줌.
    text = re.sub(r"\s+", " ", text).strip()  # 불필요한 공백 제거
    return text


def word_to_num(word, word2idx):
    try:
        return word2idx[word]  # 글자를 해당되는 정수로 변환
    except KeyError:  # 단어 집합에 없는 단어일 경우 unk로 대체된다.
        return word2idx["<unk>"]  # unk의 인덱스로 변환


def seq_padding(sequence: pd.Series):
    # 정수 인코딩된 시퀀스를 PyTorch 텐서로 변환
    encoded_tensors = [torch.tensor(seq) for seq in sequence.to_list()]
    # 패딩 적용 (최대 길이에 맞춰 0으로 패딩)
    return pad_sequence(encoded_tensors, batch_first=True, padding_value=0, padding_side="left")


def dataloader_gen(x, y, batch_size=256):
    x = torch.tensor(x, dtype=torch.int32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


LEARNING_RATE = 0.00001
N_EPOCHS = 300


EMBED_SIZE = 256
HIDDEN_SIZE = 1024
OUTPUT_DIM = 2


class SentimentAnalysisRNN(nn.Module):
    def __init__(
        self,
        vocab_dim,
        embedding_dim=EMBED_SIZE,
        hidden_dim=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM,
        device="cpu",
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x):
        embed = self.embed(x)
        y_t_list, h_t_list = self.rnn(embed)
        h_t = h_t_list.squeeze(0)

        feature = self.fc1(h_t)
        feature = F.relu(feature)
        output = self.fc2(feature)
        return self.softmax(output)


def calculate_accuracy(logits, labels):
    # _, predicted = torch.max(logits, 1)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def evaluate(model, valid_dataloader, criterion, device):
    val_loss = 0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        # 데이터로더로부터 배치 크기만큼의 데이터를 연속으로 로드
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 모델의 예측값
            logits = model(batch_X)

            # 손실을 계산
            loss = criterion(logits, batch_y)

            # 정확도와 손실을 계산함
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy
