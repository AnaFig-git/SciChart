from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import pickle
import os

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

_cached_vectorizer = None
_cached_mlp_model = None

def infer_adapter(text, model_dir='${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/'):
    global _cached_vectorizer, _cached_mlp_model
    
    if _cached_vectorizer is None or _cached_mlp_model is None:
        model_path = os.path.join(model_dir, 'instruction_adapter', 'mlp_classifier.pth')
        tokenizer_path = os.path.join(model_dir, 'instruction_adapter', 'vectorizer.pkl')

        with open(tokenizer_path, 'rb') as file:
            _cached_vectorizer = pickle.load(file)

        _cached_mlp_model = MLPClassifier(input_dim=1719, hidden_dim=512, output_dim=6)
        _cached_mlp_model.load_state_dict(torch.load(model_path))
        _cached_mlp_model.eval()

    inputs = _cached_vectorizer.transform([text]).toarray()
    inputs = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        output = _cached_mlp_model(inputs)
        _, predicted_label = torch.max(output, dim=1)

    return predicted_label.item()
