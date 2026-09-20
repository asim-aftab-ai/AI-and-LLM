# Arabic Morphological Analysis with CAMeL Tools

## Project
Arabic Morphological Analysis with CAMeL Tools

## Goal
This project uses CAMeL Tools to analyze Arabic text and explore:

* Morphological analysis
* Root extraction
* Lemma information
* Part-of-speech information
* Arabic text normalization and diacritic handling

## Environment
This project runs using the shared workspace root virtual environment (`../.venv`). No project-specific virtual environment is created.

## How to Run

From the workspace root directory:

```powershell
& ".\.venv\Scripts\streamlit.exe" run "project-06-arabic-morphology\main.py"
```

Or from within `project-06-arabic-morphology/`:

```powershell
& "..\.venv\Scripts\streamlit.exe" run main.py
```

## Architecture and Pipeline

```
Arabic Text Input
-> Tokenization (CAMeL Tools simple_word_tokenize)
-> Morphological Analysis (CAMeL Tools MorphologyDB calima-msa-r13)
-> Linguistic Feature Extraction (Root, Lemma, POS, Gender, Number, Aspect, Voice, Case, State)
-> Orthographic & Diacritic Normalization (CAMeL Tools normalization utilities)
-> Arabic-aware Representation
-> Downstream AI Pipeline (Hybrid Search / RAG / Arabic LLMs)
```
