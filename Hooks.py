"""
LINEAR PROBING WORKSHOP
Understanding What Your Transformer Actually Learned

Your trained transformer generates mostly gibberish. But did it learn NOTHING?
Today we'll use linear probes to discover what your model knows internally,
even if it can't generate good text.

What you'll learn:
1. What linear probes are and why they're useful
2. How to extract activations from transformer layers using PyTorch hooks
3. How to train probes to detect patterns
4. How information emerges across layers
"""

# %% Imports
import torch
import numpy as np
from Transformer import TransformerLanguageModel
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# %% Load Model
print("="*60)
print("LOADING MODEL")
print("="*60)

model, stoi, itos = TransformerLanguageModel.load_model('workshop_model.pt')
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
print("GENERATED TEXT (to see how bad it is)")
print("="*60)

context = torch.zeros((1, 1), dtype=torch.long)
generated = decode(model.generate(context, 200)[0].tolist())
print(generated)
print("\n" + "="*60)
print("Pretty bad, right? But let's see what it learned internally...")
print("="*60)

# %% Load Training Data
print("\n" + "="*60)
print("LOADING TRAINING DATA")
print("="*60)

with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded {len(text):,} characters")
print(f"\nFirst 200 characters:")
print(text[:200])

# %% Understanding PyTorch Hooks
print("\n" + "="*60)
print("WHAT ARE PYTORCH HOOKS?")
print("="*60)
print("""
Hooks are callback functions that PyTorch calls during forward/backward passes.
We can "hook into" any layer to capture its output without modifying the model.

Think of it like: "Hey PyTorch, whenever you compute layer 1's output,
                   save a copy for me in this variable."

This lets us see what's happening INSIDE the model during inference.
""")

# %% Simple Hook Example
print("\n" + "="*60)
print("EXAMPLE: Extracting Activations with a Hook")
print("="*60)

# Storage for activations
activations = {}

def get_activation(name):
    """Returns a hook function that stores activations"""
    def hook(module, input, output):
        # This function gets called automatically during forward pass
        activations[name] = output.detach()  # Store the output
    return hook

# Register hook on layer 1 (the last transformer block)
hook_handle = model.blocks[1].register_forward_hook(get_activation('layer_1'))

# Run a forward pass
test_text = "Hello world"
test_encoded = torch.tensor(encode(test_text), dtype=torch.long).unsqueeze(0)
_ = model(test_encoded)

# Now activations['layer_1'] contains the output!
print(f"Input text: '{test_text}'")
print(f"Input shape: {test_encoded.shape}")
print(f"Layer 1 activation shape: {activations['layer_1'].shape}")
print(f"  [batch_size, sequence_length, embedding_dim]")
print(f"  [{activations['layer_1'].shape[0]}, {activations['layer_1'].shape[1]}, {activations['layer_1'].shape[2]}]")

# Clean up hook
hook_handle.remove()

print("\n✓ Successfully extracted activations!")

# %% Prepare Dataset for Probing - Task 1: Space Detection
print("\n" + "="*60)
print("TASK 1: SPACE DETECTION")
print("="*60)
print("""
Question: Does the model know when the next character should be a space?

Even though the model generates gibberish, it might have learned that spaces
come after words. Let's probe for this!

We'll create examples where we:
1. Give the model a context (e.g., "Hello worl")
2. Label whether the next character IS a space (1) or NOT a space (0)
3. Train a linear classifier on the model's internal activations
""")

# Create dataset
def create_space_detection_dataset(text, num_samples=2000, context_len=20):
    """
    Create dataset for: "Is the next character a space?"
    
    Returns:
        contexts: List of context strings
        labels: List of 0/1 (0=not space, 1=space)
    """
    contexts = []
    labels = []
    
    for _ in range(num_samples):
        # Random position in text
        idx = np.random.randint(context_len, len(text) - 1)
        
        # Get context and next character
        context = text[idx-context_len:idx]
        next_char = text[idx]
        
        # Label: 1 if next char is space, 0 otherwise
        label = 1 if next_char == ' ' else 0
        
        contexts.append(context)
        labels.append(label)
    
    return contexts, labels

# Create dataset
contexts, labels = create_space_detection_dataset(text, num_samples=2000)

print(f"\nCreated {len(contexts)} examples")
print(f"Positive examples (next char is space): {sum(labels)}")
print(f"Negative examples (next char is NOT space): {len(labels) - sum(labels)}")

print("\nExample samples:")
for i in range(5):
    label_str = "SPACE" if labels[i] == 1 else "NOT SPACE"
    print(f"  Context: '{contexts[i][-20:]}' → {label_str}")

# %% Extract Activations for All Examples
print("\n" + "="*60)
print("EXTRACTING ACTIVATIONS")
print("="*60)
print("Running model on all examples and extracting layer activations...")

# Register hooks on ALL layers
activations = {}
hooks = []
for i, block in enumerate(model.blocks):
    hook = block.register_forward_hook(get_activation(f'layer_{i}'))
    hooks.append(hook)

# Extract activations from each layer
layer_activations = {i: [] for i in range(len(model.blocks))}

for context in contexts:
    # Encode context
    encoded = torch.tensor(encode(context), dtype=torch.long).unsqueeze(0)
    
    # Forward pass (activations are captured by hooks)
    with torch.no_grad():
        _ = model(encoded)
    
    # Store last token activation from each layer
    for i in range(len(model.blocks)):
        last_token_act = activations[f'layer_{i}'][:, -1, :].squeeze(0).numpy()
        layer_activations[i].append(last_token_act)

# Convert to numpy arrays
for i in range(len(model.blocks)):
    layer_activations[i] = np.stack(layer_activations[i])

print(f"✓ Extracted activations from {len(model.blocks)} layers")
print(f"✓ Each layer has shape: {layer_activations[0].shape} (num_examples, embed_dim)")

# Clean up hooks
for hook in hooks:
    hook.remove()

# %% Train Probes on Each Layer
print("\n" + "="*60)
print("TRAINING LINEAR PROBES")
print("="*60)
print("Training a logistic regression probe on each layer...")

labels_array = np.array(labels)
layer_accuracies = []
layer_probes = []

for layer_idx in range(len(model.blocks)):
    print(f"\nLayer {layer_idx}:")
    
    # Get activations for this layer
    X = layer_activations[layer_idx]
    y = labels_array
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train probe
    probe = LogisticRegression(max_iter=5000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    
    layer_accuracies.append(test_acc)
    layer_probes.append(probe)

# %% Visualize Results
print("\n" + "="*60)
print("RESULTS: Space Detection Across Layers")
print("="*60)

# Plot accuracy per layer
plt.figure(figsize=(10, 6))
plt.bar(range(len(layer_accuracies)), layer_accuracies, color='steelblue')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Probe Accuracy', fontsize=12)
plt.title('Space Detection: Which Layer Knows About Spaces?', fontsize=14)
plt.ylim([0, 1])
plt.axhline(y=0.5, color='r', linestyle='--', label='Random chance')
plt.xticks(range(len(layer_accuracies)))
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('space_detection_results.png', dpi=150)
print("✓ Saved plot: space_detection_results.png")
plt.show()

# Print summary
print("\nSummary:")
for i, acc in enumerate(layer_accuracies):
    print(f"  Layer {i}: {acc:.3f}")

best_layer = np.argmax(layer_accuracies)
print(f"\n✓ Best layer: Layer {best_layer} with {layer_accuracies[best_layer]:.3f} accuracy")

# %% CRITICAL DIAGNOSTIC: Is the probe actually using the activations?
print("\n" + "="*60)
print("DIAGNOSTIC: Is the probe really learning from activations?")
print("="*60)
print("""
IMPORTANT QUESTION: Are our probes actually extracting information from the
model's activations? Or are they just learning the majority class?

Let's run three tests:
1. Calculate the base rate (majority class accuracy)
2. Train a probe on RANDOM NOISE (should be ~base rate if activations don't help)
3. Compare real activations vs random noise
""")

# Test 1: Base rate
base_rate = max(sum(labels_array), len(labels_array) - sum(labels_array)) / len(labels_array)
print(f"\nTest 1 - Base Rate (majority class accuracy):")
print(f"  If we always predict the most common class: {base_rate:.3f}")

# Test 2: Probe on random noise
print(f"\nTest 2 - Training probe on RANDOM NOISE...")
random_activations = np.random.randn(len(labels_array), model.embed_dim)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    random_activations, labels_array, test_size=0.2, random_state=42, stratify=labels_array
)
probe_random = LogisticRegression(max_iter=5000, random_state=42)
probe_random.fit(X_train_rand, y_train_rand)
random_acc = probe_random.score(X_test_rand, y_test_rand)
print(f"  Probe on random noise: {random_acc:.3f}")

# Test 3: Compare
print(f"\nTest 3 - COMPARISON:")
print(f"  Base rate (always predict majority): {base_rate:.3f}")
print(f"  Probe on random noise:               {random_acc:.3f}")
print(f"  Probe on Layer 0 activations:        {layer_accuracies[0]:.3f}")
print(f"  Probe on Layer 1 activations:        {layer_accuracies[1]:.3f}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)

improvement_layer0 = layer_accuracies[0] - base_rate
improvement_layer1 = layer_accuracies[1] - base_rate

if improvement_layer0 < 0.05:  # Less than 5% improvement
    print("⚠ WARNING: Probe barely beats the base rate!")
    print("  The probe might just be learning the majority class.")
    print("  The activations may not contain much useful information.")
else:
    print(f"✓ GOOD: Probe beats base rate by {improvement_layer0:.1%}")
    print("  The activations DO contain information beyond the base rate!")
    print("  The model actually learned something about this task.")

print("\nRule of thumb:")
print("  - If probe ≈ base rate: Probe is useless, just predicting majority class")
print("  - If probe ≈ random noise: Activations don't help, no signal")
print("  - If probe >> base rate: Activations contain real information! ✓")

# Test 4: Probe weight analysis
print("\n" + "="*60)
print("BONUS TEST: Probe Weight Analysis")
print("="*60)
print("Let's check if the probe is actually using the activation dimensions...")

probe_weights = layer_probes[best_layer].coef_[0]
print(f"\nProbe weights statistics:")
print(f"  Max absolute weight:  {np.max(np.abs(probe_weights)):.4f}")
print(f"  Mean absolute weight: {np.mean(np.abs(probe_weights)):.4f}")
print(f"  Std of weights:       {np.std(probe_weights):.4f}")

# Show top 5 most important dimensions
top_dims = np.argsort(np.abs(probe_weights))[-5:][::-1]
print(f"\nTop 5 most important activation dimensions:")
for i, dim in enumerate(top_dims, 1):
    print(f"  {i}. Dimension {dim}: weight = {probe_weights[dim]:+.4f}")

print("\nIf weights are very small (< 0.01), the probe is barely using activations!")
print("="*60)

# %% Confusion Matrix for Best Layer
print("\n" + "="*60)
print(f"CONFUSION MATRIX: Layer {best_layer}")
print("="*60)

X = layer_activations[best_layer]
y = labels_array
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_probe = layer_probes[best_layer]
y_pred = best_probe.predict(X_test)

# Confusion matrix - normalized by row (shows recall per class)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Space', 'Space'],
            yticklabels=['Not Space', 'Space'],
            ax=ax1)
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title(f'Confusion Matrix (Counts): Layer {best_layer}', fontsize=12)

# Percentages (normalized)
sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', 
            xticklabels=['Not Space', 'Space'],
            yticklabels=['Not Space', 'Space'],
            ax=ax2, vmin=0, vmax=1)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_title(f'Confusion Matrix (Percentages): Layer {best_layer}', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_space.png', dpi=150)
print("✓ Saved plot: confusion_matrix_space.png")
plt.show()

print("\nReading the percentage matrix:")
print(f"  'Not Space' detection rate: {cm_normalized[0,0]:.1%} (top-left)")
print(f"  'Space' detection rate: {cm_normalized[1,1]:.1%} (bottom-right)")
print("\nIf one class has much lower percentage, the probe is biased!")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Space', 'Space']))

# %% Task 2: Capitalization Detection
print("\n" + "="*60)
print("TASK 2: CAPITALIZATION DETECTION")
print("="*60)
print("""
Question: Does the model know when the next character should be uppercase?

This tests if the model learned sentence structure, proper nouns, etc.
""")

def create_capitalization_dataset(text, num_samples=2000, context_len=20):
    """
    Create dataset for: "Should the next character be uppercase?"
    
    Returns:
        contexts: List of context strings
        labels: List of 0/1 (0=lowercase, 1=uppercase)
    """
    contexts = []
    labels = []
    
    for _ in range(num_samples):
        # Random position
        idx = np.random.randint(context_len, len(text) - 1)
        
        context = text[idx-context_len:idx]
        next_char = text[idx]
        
        # Skip if next char is not a letter
        if not next_char.isalpha():
            continue
        
        # Label: 1 if uppercase, 0 if lowercase
        label = 1 if next_char.isupper() else 0
        
        contexts.append(context)
        labels.append(label)
    
    return contexts, labels

# Create dataset
cap_contexts, cap_labels = create_capitalization_dataset(text, num_samples=2000)

print(f"\nCreated {len(cap_contexts)} examples")
print(f"Uppercase examples: {sum(cap_labels)}")
print(f"Lowercase examples: {len(cap_labels) - sum(cap_labels)}")

print("\nExample samples:")
for i in range(5):
    if i >= len(cap_contexts):
        break
    label_str = "UPPERCASE" if cap_labels[i] == 1 else "lowercase"
    print(f"  Context: '{cap_contexts[i][-20:]}' → {label_str}")

# %% Extract Activations for Capitalization Task
print("\n" + "="*60)
print("EXTRACTING ACTIVATIONS FOR CAPITALIZATION")
print("="*60)

# Register hooks
activations = {}
hooks = []
for i, block in enumerate(model.blocks):
    hook = block.register_forward_hook(get_activation(f'layer_{i}'))
    hooks.append(hook)

# Extract activations
cap_layer_activations = {i: [] for i in range(len(model.blocks))}

for context in cap_contexts:
    encoded = torch.tensor(encode(context), dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        _ = model(encoded)
    
    for i in range(len(model.blocks)):
        last_token_act = activations[f'layer_{i}'][:, -1, :].squeeze(0).numpy()
        cap_layer_activations[i].append(last_token_act)

# Convert to arrays
for i in range(len(model.blocks)):
    cap_layer_activations[i] = np.stack(cap_layer_activations[i])

print(f"✓ Extracted activations")

# Clean up hooks
for hook in hooks:
    hook.remove()

# %% Train Capitalization Probes
print("\n" + "="*60)
print("TRAINING CAPITALIZATION PROBES")
print("="*60)

cap_labels_array = np.array(cap_labels)
cap_layer_accuracies = []

for layer_idx in range(len(model.blocks)):
    print(f"\nLayer {layer_idx}:")
    
    X = cap_layer_activations[layer_idx]
    y = cap_labels_array
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train probe
    probe = LogisticRegression(max_iter=5000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    
    cap_layer_accuracies.append(test_acc)

# %% Visualize Capitalization Results
print("\n" + "="*60)
print("RESULTS: Capitalization Detection Across Layers")
print("="*60)

plt.figure(figsize=(10, 6))
plt.bar(range(len(cap_layer_accuracies)), cap_layer_accuracies, color='coral')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Probe Accuracy', fontsize=12)
plt.title('Capitalization Detection: Which Layer Knows About Uppercase?', fontsize=14)
plt.ylim([0, 1])
plt.axhline(y=0.5, color='r', linestyle='--', label='Random chance')
plt.xticks(range(len(cap_layer_accuracies)))
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('capitalization_results.png', dpi=150)
print("✓ Saved plot: capitalization_results.png")
plt.show()

print("\nSummary:")
for i, acc in enumerate(cap_layer_accuracies):
    print(f"  Layer {i}: {acc:.3f}")

best_cap_layer = np.argmax(cap_layer_accuracies)
print(f"\n✓ Best layer: Layer {best_cap_layer} with {cap_layer_accuracies[best_cap_layer]:.3f} accuracy")

# %% Confusion Matrix for Capitalization (Best Layer)
print("\n" + "="*60)
print(f"CONFUSION MATRIX: Capitalization - Layer {best_cap_layer}")
print("="*60)

X = cap_layer_activations[best_cap_layer]
y = cap_labels_array
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train probe for best layer
cap_best_probe = LogisticRegression(max_iter=5000, random_state=42)
cap_best_probe.fit(X_train, y_train)
y_pred = cap_best_probe.predict(X_test)

# Confusion matrix - normalized by row
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Lowercase', 'Uppercase'],
            yticklabels=['Lowercase', 'Uppercase'],
            ax=ax1)
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title(f'Confusion Matrix (Counts): Layer {best_cap_layer}', fontsize=12)

# Percentages
sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Oranges', 
            xticklabels=['Lowercase', 'Uppercase'],
            yticklabels=['Lowercase', 'Uppercase'],
            ax=ax2, vmin=0, vmax=1)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_title(f'Confusion Matrix (Percentages): Layer {best_cap_layer}', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_capitalization.png', dpi=150)
print("✓ Saved plot: confusion_matrix_capitalization.png")
plt.show()

print("\nReading the percentage matrix:")
print(f"  'Lowercase' detection rate: {cm_normalized[0,0]:.1%} (top-left)")
print(f"  'Uppercase' detection rate: {cm_normalized[1,1]:.1%} (bottom-right)")
print("\nIf one class has much lower percentage, the probe is biased!")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Lowercase', 'Uppercase']))

# %% DIAGNOSTIC: Capitalization probe validation
print("\n" + "="*60)
print("DIAGNOSTIC: Is the capitalization probe really learning?")
print("="*60)

# Test 1: Base rate
cap_base_rate = max(sum(cap_labels_array), len(cap_labels_array) - sum(cap_labels_array)) / len(cap_labels_array)
print(f"\nTest 1 - Base Rate (majority class accuracy):")
print(f"  If we always predict the most common class: {cap_base_rate:.3f}")

# Test 2: Probe on random noise
print(f"\nTest 2 - Training probe on RANDOM NOISE...")
random_cap_activations = np.random.randn(len(cap_labels_array), model.embed_dim)
X_train_rand_cap, X_test_rand_cap, y_train_rand_cap, y_test_rand_cap = train_test_split(
    random_cap_activations, cap_labels_array, test_size=0.2, random_state=42, stratify=cap_labels_array
)
probe_random_cap = LogisticRegression(max_iter=5000, random_state=42)
probe_random_cap.fit(X_train_rand_cap, y_train_rand_cap)
random_cap_acc = probe_random_cap.score(X_test_rand_cap, y_test_rand_cap)
print(f"  Probe on random noise: {random_cap_acc:.3f}")

# Test 3: Compare
print(f"\nTest 3 - COMPARISON:")
print(f"  Base rate (always predict majority): {cap_base_rate:.3f}")
print(f"  Probe on random noise:               {random_cap_acc:.3f}")
print(f"  Probe on Layer 0 activations:        {cap_layer_accuracies[0]:.3f}")
print(f"  Probe on Layer 1 activations:        {cap_layer_accuracies[1]:.3f}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)

cap_improvement_layer0 = cap_layer_accuracies[0] - cap_base_rate
cap_improvement_layer1 = cap_layer_accuracies[1] - cap_base_rate

if cap_improvement_layer0 < 0.05:  # Less than 5% improvement
    print("⚠ WARNING: Probe barely beats the base rate!")
    print("  The probe might just be learning the majority class.")
    print("  The activations may not contain much useful information about capitalization.")
else:
    print(f"✓ GOOD: Probe beats base rate by {cap_improvement_layer0:.1%}")
    print("  The activations DO contain information about capitalization!")

print("\n" + "="*60)

# %% Compare Both Tasks
print("\n" + "="*60)
print("COMPARISON: Both Tasks Across Layers")
print("="*60)

plt.figure(figsize=(12, 6))
x = range(len(layer_accuracies))
width = 0.35

plt.bar([i - width/2 for i in x], layer_accuracies, width, label='Space Detection', color='steelblue')
plt.bar([i + width/2 for i in x], cap_layer_accuracies, width, label='Capitalization', color='coral')

plt.xlabel('Layer', fontsize=12)
plt.ylabel('Probe Accuracy', fontsize=12)
plt.title('Comparing Probing Tasks Across Layers', fontsize=14)
plt.ylim([0, 1])
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random chance')
plt.xticks(x)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_both_tasks.png', dpi=150)
print("✓ Saved plot: comparison_both_tasks.png")
plt.show()

# %% Key Insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
What did we learn?

1. Even though the model generates bad text, it DID learn internal structure!
   - Space detection works well (likely > 70% accuracy)
   - Capitalization detection also works (depends on training)

2. Information emerges across layers:
   - Compare layer 0 vs layer 1 accuracies
   - Which layer is better at each task?
   - Does information get refined as it goes deeper?

3. Linear probes reveal hidden knowledge:
   - The model "knows" things internally it can't express in generation
   - Probing shows what representations exist, even if generation fails

4. Always validate your probes!
   - Compare probe accuracy to base rate (majority class)
   - Test on random noise to ensure activations matter
   - Check if probe weights are meaningful
   - Confusion matrices reveal class-specific performance
   
5. High accuracy doesn't always mean success:
   - A probe might just learn the majority class
   - Class imbalance can make results misleading
   - Always look beyond overall accuracy!

6. This is how we do mechanistic interpretability:
   - We don't just look at outputs
   - We peer inside and ask: "what does each layer represent?"
   - We validate that our methods are actually measuring something real
""")

# %% Exercises for Students
print("\n" + "="*60)
print("EXERCISES FOR YOU TO TRY")
print("="*60)
print("""
1. Run diagnostics on the capitalization task:
   - Calculate base rate for uppercase vs lowercase
   - Train probe on random noise
   - Is capitalization probe really learning from activations?
   - Why might results differ from space detection?

2. Create a new probing task:
   - Probe for vowels vs consonants
   - Probe for punctuation detection
   - Probe for word boundaries (is this the last character of a word?)

3. Experiment with probe complexity:
   - Try a simple Linear SVM instead of Logistic Regression
   - Does it change accuracy?

4. Visualize probe weights:
   - Look at probe.coef_ to see which dimensions the probe uses
   - Which embedding dimensions are most important?
   - Are they consistent across layers?

5. Probe intermediate components:
   - Instead of probing full blocks, probe just the attention output
   - Or just the MLP output
   - Register hooks on model.blocks[i].attn or model.blocks[i].ff

6. Error analysis:
   - For the best layer, look at examples the probe got wrong
   - What patterns does it miss?
   - Print contexts where predictions failed
   
7. Class imbalance experiments:
   - What happens if you balance your dataset (equal positives/negatives)?
   - Does probe accuracy change?
   - Does the confusion matrix look different?
""")

print("\n" + "="*60)
print("WORKSHOP COMPLETE!")
print("="*60)
print("\nYou've learned:")
print("✓ How to extract activations with PyTorch hooks")
print("✓ How to train linear probes")
print("✓ How to validate probes (base rate, random noise, weight analysis)")
print("✓ How to interpret confusion matrices (counts AND percentages)")
print("✓ How to compare learning across layers")
print("✓ That models can have rich internal knowledge even with bad outputs")
print("✓ That high accuracy can be misleading - always check for bias!")
print("\nNext steps: Try the exercises above and explore your model further!")