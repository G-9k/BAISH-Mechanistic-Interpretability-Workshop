# Mechanistic Interpretability Workshop: Part 1
**Uncovering the Hidden Knowledge of Toy Transformers**

Welcome to the Mechanistic Interpretability workshop! In this first session, we will dive into the internal representations of a "Toy Transformer." Even though small language models often generate incoherent text, they frequently learn sophisticated internal features about language structure (syntax, grammar, and boundaries).

We will use **Linear Probes** and **PyTorch Hooks** to surgically extract these activations and prove what the model knows, layer by layer.

## 🧠 What is a Linear Probe?

A linear probe is a diagnostic tool used to understand deep learning models. Think of it like a "mind-reading" device for neural networks.

1.  **The Premise:** As data moves through the layers of a Transformer, the model transforms it into high-dimensional vectors (embeddings).
2.  **The Hypothesis:** If the model understands a concept (like "the current word is a verb"), that information must be encoded linearly within those vectors.
3.  **The Method:** We freeze the model. We extract the internal activation vectors and train a simple linear classifier (like Logistic Regression) to predict a specific feature (e.g., "Is this a space?").
4.  **The Result:** If the simple classifier achieves high accuracy, we know the model has explicitly learned and represented that feature at that specific layer.

## 🛠️ Installation & Setup

### 1. Prerequisites
You will need:
* **Python 3.8+**
* A code editor that supports **Interactive Python Cells** (highly recommended).
    * **VS Code** (install the Python extension)
    * **Spyder** or **JupyterLab**
    * *Note: The script uses `# %%` cell markers, which allows you to run code in blocks like a notebook.*

### 2. Installation
Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## 🚀 How to Follow the Workshop
Open ```Linear_Probes.py```. This file is designed to be run **interactively**.

### Recommended Workflow (VS Code):

1. Open ```Linear_Probes.py```.
2. You will see "Run Cell" | "Run Below" Options appear above the ```# %%``` comments.
3. Click "**Run Cell** to execute one block at a time.
4. Watch the output in the Interactive Window on the right.

### The Workshop Activities
The script guides you through four main stages:

1.  **The Hook:** We will write a custom PyTorch Hook to intercept the data flowing through the model's layers without modifying the model source code.
2. **Task 1 - Space Detection:** We will extract activations and train a probe to see if the model knows where spaces belong.
3. **Task 2 - Capitalization:** We will repeat the process for capital letters.
4. **Diagnostics:** We will challenge our findings. Is the probe actually learning, or it is just guessing? We will compare our results against a "Random Noise" baseline.

## 🧪 Exercises

At the bottom of the script, there is an "Exercises" section. Try to complete at least:

* **Exercise 1 (Diagnostics):** Validate the capitalization probe

* **Exercise 2 (New Taks):** Write a probe for Vowels vs. Consonants.

## 🔜 Coming Soon: Part 2

In the next session, we will move away from raw PyTorch hooks and introduce **Transformer Lens**, a library specifically designed to make mechanistic interpretability easier, faster, and more powerful.