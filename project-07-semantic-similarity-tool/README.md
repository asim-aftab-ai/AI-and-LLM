# Project 07: Semantic Similarity Tool

A minimalist, beginner-friendly application comparing the semantic meaning of two pieces of text using sentence embeddings and cosine similarity.

---

## 1. What the Project Does

This application takes two text inputs, transforms them into high-dimensional numerical vectors using a local sentence embedding model (`all-MiniLM-L6-v2`), and measures how closely their underlying meanings relate to each other on a normalized scale from **0.00 to 1.00**.

---

## 2. Core Concepts

### What is Semantic Similarity?
Unlike traditional keyword search that counts character or word overlap, **semantic similarity** measures whether two sentences convey the same meaning. For example:
- *Text A*: "How can I renew my residence permit?"
- *Text B*: "What is the process for extending a residence visa?"

These two sentences share almost no identical content words, yet a human understands they ask the same question. Semantic similarity models quantify this conceptual equivalence.

### What are Embeddings?
An **embedding** is a translation of text into a vector of numbers (e.g. 384 dimensions). Neural language models position sentences with similar meanings close together in this vector space, while unrelated sentences are positioned far apart.

### What is Cosine Similarity?
**Cosine similarity** measures the cosine of the angle between two multi-dimensional vectors:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- Value near **1.0**: The vectors point in virtually identical directions (very high similarity).
- Value near **0.0**: The vectors are perpendicular/orthogonal (unrelated concepts).

---

## 3. Conceptual Pipeline

```text
Text A                     Text B
   ↓                          ↓
Embedding Model            Embedding Model
(`all-MiniLM-L6-v2`)       (`all-MiniLM-L6-v2`)
   ↓                          ↓
384-d Vector A             384-d Vector B
        \                     /
         \                   /
          Cosine Similarity(A, B)
                     ↓
             Score (0.00 – 1.00)
                     ↓
             Human Explanation
```

---

## 4. Score Interpretation

* **0.80 – 1.00:** Very High Semantic Similarity
* **0.60 – 0.79:** High Semantic Similarity
* **0.40 – 0.59:** Moderate Semantic Similarity
* **0.20 – 0.39:** Low Semantic Similarity
* **0.00 – 0.19:** Very Low Semantic Similarity

---

## 5. How to Run

From the workspace root directory:

```powershell
& ".\.venv\Scripts\streamlit.exe" run "project-07-semantic-similarity-tool\app.py"
```

Or from inside `project-07-semantic-similarity-tool/`:

```powershell
& "..\.venv\Scripts\streamlit.exe" run app.py
```

---

## 6. Main Dependencies

* `streamlit`: Minimalist web interface.
* `transformers`: Model architecture and tokenization.
* `torch`: PyTorch runtime for tensor operations and cosine similarity.
