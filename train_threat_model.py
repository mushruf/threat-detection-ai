import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Load dataset (replace with real logs)
data = {
    "log_message": [
        "Failed login attempt from IP 192.168.1.1",
        "Suspicious activity detected in user session",
        "Multiple failed login attempts detected",
        "Unauthorized SSH access attempt",
        "Normal user login event",
        "Brute-force attack detected on web server"
    ],
    "threat_level": ["low", "medium", "high", "high", "low", "critical"]
}
df = pd.DataFrame(data)

# ✅ Preprocessing
label_encoder = LabelEncoder()
df["threat_level"] = label_encoder.fit_transform(df["threat_level"])

# ✅ Convert logs into numeric vectors (simple encoding)
vectorizer = {word: idx for idx, word in enumerate(set(" ".join(df["log_message"]).split()))}
df["log_vector"] = df["log_message"].apply(lambda x: [vectorizer[word] for word in x.split()])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["log_vector"], df["threat_level"], test_size=0.2, random_state=42)

# ✅ Convert to tensors
def pad_sequences(seq_list, max_len=10):
    return [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in seq_list]

X_train, X_test = pad_sequences(X_train), pad_sequences(X_test)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train.values, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long)

# ✅ Define Model
class ThreatDetectionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=16, output_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# ✅ Train Model
vocab_size = len(vectorizer) + 1
model = ThreatDetectionModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ✅ Training Loop
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# ✅ Save model
torch.save(model.state_dict(), "threat_detection_model.pt")
print("Model saved successfully!")
