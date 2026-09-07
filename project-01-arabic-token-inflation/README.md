# English vs Arabic Tokenization & Token Inflation

An interactive, educational GenAI learning tool to analyze and compare how subword tokenizers represent English versus Arabic text, inspect subword divisions, and measure the empirical **Arabic Token Inflation Factor**.

---

## 🎯 Objective

Large Language Models (LLMs) do not read text character by character or word by word — they process **tokens**. Because modern foundational LLM tokenizers (such as OpenAI's Byte-Pair Encoding) were predominantly trained on English-heavy web datasets, non-Latin scripts such as Arabic often undergo different subword fragmentation.

This project empirically compares:
1. **English tokenization** vs **Arabic tokenization** on semantically equivalent texts.
2. **Subword splits** (how words are segmented into computational chunks).
3. **Token counts** and resulting **Arabic Token Inflation Factors**.
4. **General-purpose GPT-style tokenizers (`cl100k_base`)** vs **Arabic-dedicated tokenizers (AraBERT)**.
5. The real-world consequences on **LLM API costs** and **context window capacities**.

---

## 🛠️ Tools Used

- **Python 3:** Core programming language and runtime.
- **tiktoken:** OpenAI's fast BPE tokenizer library, using the `cl100k_base` encoding (the encoding powering GPT-4 and GPT-3.5-turbo).
- **Hugging Face Transformers / Tokenizers:** Provides `AutoTokenizer` to load `aubmindlab/bert-base-arabertv02`, an Arabic-focused BERT tokenizer with 64k vocabulary trained specifically on Arabic corpora.
- **Streamlit:** Fast, interactive web application framework to explore inputs, visualize subwords, and view dynamic metrics in the browser.
- **Altair & Pandas:** Data modeling and interactive charting.

---

## 💡 Important Concepts

### 1. Token
A **token** is the fundamental atomic unit of text that a language model reads and generates. It can be a whole word, a subword fragment, a single character, or a byte sequence.

### 2. Tokenizer
A software component or algorithm (such as Byte-Pair Encoding or WordPiece) that converts human-readable strings into numeric token IDs (integers) that neural networks can process.

### 3. Subword
A fractional piece of a word. When a full word is not in a tokenizer's predefined vocabulary, the tokenizer breaks the word into multiple smaller subwords (e.g., `"tokenization"` might become `["token", "ization"]`).

### 4. Token Count
The total number of tokens produced by encoding a given piece of text.

### 5. Arabic Token Inflation Factor
The quantitative ratio measuring relative token usage between semantically equivalent texts:
$$\text{Arabic Token Inflation Factor} = \frac{\text{Arabic Token Count}}{\text{English Token Count}}$$

- **1.0x:** Parity (both versions require approximately the same token budget).
- **Above 1.0x:** Arabic uses more tokens for that specific sentence and tokenizer.
- **Below 1.0x:** Arabic uses fewer tokens for that comparison.

### 6. Context Window
The strict upper bound of tokens an LLM can accept in a single prompt + response cycle (e.g., 8k, 32k, 128k). High token counts fill the context window faster, reducing the number of conversational turns or document length a model can remember.

### 7. LLM Cost
Commercial LLM APIs charge **per million tokens**. If an Arabic translation of a prompt uses 1.6x more tokens than the English original, you pay 60% more for the exact same semantic task.

---

## 🇸🇦 Arabic Linguistic Considerations

Arabic is not a uniform, monolithic input type. The behavior of a tokenizer is heavily influenced by:

1. **Modern Standard Arabic (MSA):** The formal written Arabic used across official institutions, news, and banking.
2. **Dialect Variations (e.g., Gulf/Khaliji, Egyptian, Levantine, Maghrebi):** Colloquial dialects have unique vocabularies, grammatical simplifications, and regional spellings that are rare in standard training corpora, often causing higher subword fragmentation.
3. **Arabizi (Franco-Arabic):** Arabic phonetically written in Latin characters combined with numerals (e.g., `"shokran"`, `"3arab"`), which creates unique, non-standard subwords.
4. **Morphology & Concatenation:** Arabic is a root-and-pattern language. Conjunctions (`و` = "and"), prepositions (`ب` = "by/with", `ل` = "for"), and definite articles (`ال` = "the") attach directly to the root noun as prefixes. If a tokenizer's vocabulary does not contain the prefixed form, it must break the word into multiple tokens.
5. **Orthographic Normalization & Diacritics (Tashkeel):** Short vowel marks (tashkeel) add extra characters that standard tokenizers often split into separate bytes, drastically inflating token counts.

> **Important Principle:** We do **not** assume that Arabic is "always twice as expensive." This project is designed to **measure** actual empirical differences across distinct tokenizers and diverse texts.

---

## 🏃 How to Run

### Step 1: Activate the Virtual Environment

From the root `AI and LLM` folder in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies (if not already installed)

```powershell
pip install -r project-01-arabic-token-inflation\requirements.txt
```

### Step 3: Launch the Streamlit App

```powershell
cd project-01-arabic-token-inflation
streamlit run app.py
```

Streamlit will print the local URL:
```text
Local URL: http://localhost:8501
```
Open this URL in your web browser to start experimenting!
