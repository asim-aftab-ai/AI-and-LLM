# Project 06: Arabic NLP & Bilingual Text Processing

This project provides hands-on educational tools for exploring Arabic Natural Language Processing with a comprehensive **Bilingual (Arabic + English)** learning layer. The English layer serves as a learning aid so students and developers who are beginning to learn Arabic can fully understand the semantic meaning of Arabic texts, summaries, and morphological structures.

---

## Workspace & Environment

This project runs using the shared workspace root virtual environment (`../.venv`). No additional or project-specific virtual environments are created.

### Direct Dependencies

- `camel-tools`: Modern Standard Arabic morphological analysis, word tokenization, and orthographic normalization.
- `streamlit`: Interactive Arabic morphological analysis laboratory.
- `gradio`: Interactive bilingual text summarizer interface (`gr.Interface`, `gr.Textbox`, `gr.Button`, `share=True`).
- `deep-translator`: High-accuracy Arabic-to-English translation layer providing learning glosses and translations.

---

## Application 1: Arabic Morphological Analysis (Streamlit)

A technical laboratory exploring Arabic root-and-pattern morphology, tokenization, and normalization with integrated bilingual explanations:

* **Sentence & Token Meanings**: Displays live English interpretations alongside Arabic inputs and segmented tokens.
* **Morphological Features**: Exposes dictionary lemmas (`lex`), grammatical categories (`pos`), trilateral roots (`root`), and inflectional features (gender, number, person, aspect, case, and state) with clear English glosses.
* **Polysemy & Ambiguity**: Demonstrates why a single Arabic surface token can yield multiple distinct valid morphological interpretations.
* **Normalization Engine**: Demonstrates dediacritization and character standardization (Alef, Teh Marbuta, Alef Maksura).

### How to Run

From the workspace root:
```powershell
& ".\.venv\Scripts\streamlit.exe" run "project-06-arabic-morphology\main.py"
```

Or from inside `project-06-arabic-morphology/`:
```powershell
& "..\.venv\Scripts\streamlit.exe" run main.py
```

---

## Application 2: Bilingual Text Summarizer (Gradio)

An educational extractive summarization interface with side-by-side English translations to help language learners understand both the source and the condensed summary.

### Bilingual Conceptual Layout

```
-----------------------------------------
TEXT SUMMARIZER (Bilingual)
-----------------------------------------
Original Arabic / Text:
[ Arabic text here ]

English Meaning:
[ English translation of original text ]
-----------------------------------------
SUMMARY:
Arabic Summary:
[ Extractive Arabic summary ]

English Meaning of Summary:
[ English translation corresponding to the Arabic summary ]
-----------------------------------------
STATISTICS:
Original Word Count: XX
Summary Word Count: XX
Compression Ratio: XX%
-----------------------------------------
```

### Extractive Summarization Logic

1. **Sentence Boundary Detection**: Segments text across Arabic (`.`, `؟`, `!`) and English (`.`, `?`, `!`) sentence boundaries.
2. **Frequency Scoring**: Computes normalized word frequencies while filtering non-content stopwords.
3. **Sentence Scoring & Length Penalty**: Ranks sentences based on constituent term significance normalized by sentence length.
4. **Order-Preserving Assembly**: Selects top sentences and restores chronological order.
5. **Exact English Correspondence**: Translates the extractive Arabic summary so the learner can verify the summary's meaning.
6. **Compression Metric**:
   $$\text{Compression Ratio} = \left(\frac{\text{Original Words} - \text{Summary Words}}{\text{Original Words}}\right) \times 100$$
   Safely handles empty or single-sentence inputs without division-by-zero errors.

### Gradio Interface Features

- `gr.Interface`: Application container.
- `gr.Textbox`: Multi-line Arabic text input and distinct English translation/statistic outputs.
- `gr.Button`: Primary "Summarize" trigger button.
- `share=True`: Generates both local access and a public `gradio.live` link for learning demonstration.

### How to Run

From the workspace root:
```powershell
& ".\.venv\Scripts\python.exe" "project-06-arabic-morphology\gradio_app.py"
```

Or from inside `project-06-arabic-morphology/`:
```powershell
& "..\.venv\Scripts\python.exe" gradio_app.py
```
