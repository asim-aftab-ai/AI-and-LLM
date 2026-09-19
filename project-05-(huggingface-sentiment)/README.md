# Project 05: Hugging Face Sentiment Analysis

## Project Purpose
This project is a clean, focused demonstration of using Hugging Face Transformers for sentiment analysis within an interactive Streamlit web application. It evaluates exactly 10 predefined English sentences covering positive, negative, and neutral/factual statements, showing the predicted sentiment label and confidence score for each.

## What Sentiment Analysis Is
Sentiment analysis is a subfield of Natural Language Processing (NLP) that aims to determine the emotional orientation, attitude, or tone conveyed in natural language text. In binary classification setups, sentiment models typically identify whether a statement expresses positive or negative emotion, accompanied by a probability or confidence score indicating the model's certainty.

## What Hugging Face Is Being Used For
The project leverages the Hugging Face Transformers library, specifically using its high-level `pipeline("sentiment-analysis")` task. By default, this pipeline downloads and runs `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, a distilled BERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset. The pipeline handles tokenization, tensor conversion, neural model inference, and output label mapping. The model instance is cached upon initial load so inference runs efficiently without reloading weights repeatedly.

## What the Streamlit Frontend Does
The Streamlit frontend (`main.py`) provides an intuitive, web-based graphical interface:
- Displays a clear overview and description of sentiment analysis.
- Presents the list of 10 predefined sentences to be evaluated.
- Offers an "Analyze Sentiment" button to trigger the analysis on demand.
- Shows a loading indicator while the model performs inference.
- Presents the final results in a structured table containing the original sentence, predicted sentiment label (POSITIVE or NEGATIVE), and confidence score formatted as a percentage.
- Gracefully handles and displays errors if model loading or inference encounters issues.

## Project Structure
```
project-05-(huggingface-sentiment)/
│
├── app.py                  # Standard entrypoint (delegates to main)
├── main.py                 # Interactive Streamlit frontend UI
├── sentiment_analyzer.py   # Hugging Face DistilBERT inference module
├── sentences.py            # Canonical 10 benchmark sentences
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## How the Files Communicate
The system follows a clean separation of concerns across its modules:
1. `sentences.py` defines the canonical list of 10 predefined English sentences and exposes `get_predefined_sentences()`.
2. `main.py` imports `get_predefined_sentences` from `sentences.py` to display the initial sentences on the UI.
3. When the user clicks "Analyze Sentiment" in `main.py`, the sentences are passed to `analyze_sentences()` in `sentiment_analyzer.py`.
4. `sentiment_analyzer.py` invokes the cached Hugging Face pipeline, processes the raw model predictions, and returns structured dictionaries containing the sentence, sentiment label, and numerical score.
5. `main.py` formats the output scores as percentages and renders them in a clear table on the web page.

## Architectural Flow
```
10 predefined sentences
        |
   sentences.py
        |
     main.py
        |
sentiment_analyzer.py
        |
Hugging Face sentiment-analysis pipeline
        |
  Prediction results
        |
     main.py
        |
   Streamlit UI
        |
Label + confidence score for each sentence
```

## Required Dependencies
- `streamlit`
- `transformers`
- `torch`

These dependencies are recorded in `requirements.txt`.

## Running the Application
The workspace uses a shared root-level virtual environment located at `.venv`.

To run the application using this existing virtual environment:

1. Open a terminal in the root workspace directory or navigate into the project directory:
```bash
cd project-05-(huggingface-sentiment)
```

2. Launch Streamlit using the root `.venv` Python executable:
```bash
..\.venv\Scripts\streamlit run app.py
```
*(You can also use `streamlit run main.py`)*

Alternatively, from the workspace root:
```bash
.venv\Scripts\streamlit run project-05-(huggingface-sentiment)/app.py
```

## What to Expect
When running the Streamlit app:
1. A browser window will open at `http://localhost:8501`.
2. The page displays the title "Hugging Face Sentiment Analysis" and model overview metrics.
3. In the **Benchmark** tab, all 10 sentences are listed in order.
4. Clicking "Analyze Sentiment" executes the inference pipeline and displays summary statistics and an interactive table with colored sentiment badges and percentage confidence scores.
5. In the **Interactive Playground** tab, you can enter any custom sentence or pick a sample preset to test real-time sentiment inference.
6. The **Deep Dive** tab explains why binary classification (SST-2) assigns positive/negative to neutral sentences and how DistilBERT distillation functions.

