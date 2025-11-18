import torch
import numpy as np
from Transformer import TransformerLanguageModel, estimate_loss
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# %% Load Model
print("="*60)
print("LOADING MODEL")
print("="*60)

model, stoi, itos = TransformerLanguageModel.load_model('Transformer1.pt')
model.eval()

# Helper functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"\nModel info:")
print(f"  Number of layers: {len(model.blocks)}")
print(f"  Embedding dimension: {model.embed_dim}")
print(f"  Vocabulary size: {len(stoi)}")

# %% Check Model Output
print("\n" + "="*60)
print("GENERATED TEXT With layer 1:")
print("="*60)

context = torch.zeros((1, 1), dtype=torch.long)
generated = decode(model.generate(context, 500)[0].tolist())
print(generated)
print("\n" + "="*60)

# Zero out layer 1's contribution
def ablate_layer_1(model):
    # Save original forward
    original_forward = model.blocks[1].forward
    
    # Replace with identity function (pass through unchanged)
    def identity_forward(x):
        return x
    
    model.blocks[1].forward = identity_forward
    return model, original_forward

# Test generation quality with/without layer 1
model_ablated, original = ablate_layer_1(model)
text_without_layer1 = decode(model_ablated.generate(context, 500)[0].tolist())
print("Without layer 1:")
print(text_without_layer1)
