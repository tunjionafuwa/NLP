from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data = [
    (["John", "lives", "in", "New", "York"], ["B-PER", "O", "O", "B-LOC", "I-LOC"]),
    (["Mary", "works", "at", "Google"], ["B-PER", "O", "O", "B-ORG"]),
    (["Berlin", "is", "the", "capital", "of", "Germany"], ["B-LOC", "O", "O", "O", "O", "B-LOC"]),
]

word2idx = defaultdict(lambda: len(word2idx))
tag2idx = defaultdict(lambda: len(tag2idx))
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

for sentences, tags in data:
    for sentence in sentences:
        word2idx[sentence]
    for tag in tags:
        tag2idx[tag]

idx2tag = {i: t for t, i in tag2idx.items()}


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        x = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
        y = [tag2idx[t] for t in tags]
        return torch.tensor(x), torch.tensor(y), len(x)

def pad_batch(batch):
    xs, ys, lens = zip(*batch)
    max_len = max(lens)
    x_pad = torch.zeros(len(xs), max_len).long()
    y_pad = torch.zeros(len(xs), max_len).long()
    for i in range(len(xs)):
        x_pad[i, :lens[i]] = xs[i]
        y_pad[i, :lens[i]] = ys[i]
    return x_pad, y_pad, torch.tensor(lens)

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx["<PAD>"])
        self.lstm = nn.RNN(embedding_dim, hidden_dim // 2, num_layers=4, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(lstm_out)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = NERDataset(data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_batch, shuffle=True)

model = BiLSTM_NER(vocab_size=len(word2idx), tagset_size=len(tag2idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    model.train()
    for x_batch, y_batch, lengths in dataloader:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x_batch, lengths)
        loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
def predict(sentence):
    tokens = sentence.split()
    x = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in tokens]).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)]).to(device)
    with torch.no_grad():
        logits = model(x, lengths)
        pred = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        return list(zip(tokens, [idx2tag.get(p, "O") for p in pred]))

# Example usage
print(predict("Alice lives in Paris"))
