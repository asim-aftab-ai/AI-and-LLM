"""
Tokenizer Analysis Module: English vs Arabic
=============================================
This module provides core analysis functions to compare token counts,
subword segmentations, and token inflation factors between English and
Arabic text using OpenAI's tiktoken (cl100k_base) and Hugging Face's
AraBERT tokenizer (aubmindlab/bert-base-arabertv02).

Author: Antigravity GenAI Learning Series
Project: 01 - English vs Arabic Token Inflation
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Built-in Paired Datasets (Banking & News)
# ---------------------------------------------------------------------------
# Paired examples with equivalent semantic meanings for accurate comparison.
SAMPLE_DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "Banking": {
        "Account Opening": {
            "english": "I would like to open a new savings account with your bank today.",
            "arabic": "أود فتح حساب توفير جديد لدى بنككم اليوم."
        },
        "Credit Card Issue": {
            "english": "My credit card was blocked because I entered the wrong PIN three times.",
            "arabic": "تم حظر بطاقتي الائتمانية لأنني أدخلت الرمز السري بشكل خاطئ ثلاث مرات."
        },
        "Suspicious Charge": {
            "english": "I noticed an unauthorized charge of five hundred dollars on my statement.",
            "arabic": "لاحظت عملية خصم غير مصرح بها بقيمة خمسمائة دولار في كشف حسابي."
        },
        "Loan Inquiry": {
            "english": "What are the interest rates and repayment terms for a personal loan?",
            "arabic": "ما هي أسعار الفائدة وشروط السداد للحصول على قرض شخصي؟"
        },
        "Balance Check": {
            "english": "Please send me my current account balance and recent transactions.",
            "arabic": "يرجى إرسال رصيد حسابي الحالي وقائمة المعاملات الأخيرة."
        }
    },
    "News": {
        "Economic Policy": {
            "english": "The central bank decided to lower benchmark interest rates to stimulate economic growth.",
            "arabic": "قرر البنك المركزي خفض أسعار الفائدة الرئيسية لتحفيز النمو الاقتصادي."
        },
        "Tech Investment": {
            "english": "Artificial intelligence startups in the region attracted record venture capital investments.",
            "arabic": "استقطبت الشركات الناشئة في مجال الذكاء الاصطناعي في المنطقة استثمارات قياسية من رأس المال الجريء."
        },
        "Business Expansion": {
            "english": "The international retail company announced plans to expand its branches across the Middle East.",
            "arabic": "أعلنت شركة التجزئة الدولية عن خطط لتوسيع فروعها في جميع أنحاء الشرق الأوسط."
        },
        "Energy Markets": {
            "english": "Global oil markets witnessed increased volatility amid fluctuating supply and demand.",
            "arabic": "شهدت أسواق النفط العالمية تقلبات متزايدة وسط تذبذب في العرض والطلب."
        }
    }
}


# ---------------------------------------------------------------------------
# 2. Tiktoken Analyzer (cl100k_base)
# ---------------------------------------------------------------------------
def analyze_with_tiktoken(text: str, encoding_name: str = "cl100k_base") -> Dict[str, Any]:
    """
    Analyzes input text using OpenAI's tiktoken library.
    
    Parameters:
        text: The text string to tokenize.
        encoding_name: Name of the byte-pair encoding.
                       'cl100k_base' is standard for GPT-4, GPT-3.5-turbo, etc.
                       
    Returns:
        Dict containing:
            - encoding_name: Name of encoding used
            - token_count: Total number of tokens
            - token_ids: Raw integer token IDs
            - subwords: Decoded string representation of each individual token
            - byte_lengths: Number of bytes per token
    """
    if not text.strip():
        return {
            "encoding_name": encoding_name,
            "token_count": 0,
            "token_ids": [],
            "subwords": [],
            "byte_lengths": [],
            "error": None
        }

    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        token_ids = enc.encode(text)
        
        # Safely extract subword representations without failing on UTF-8 multi-byte splits
        subwords = []
        byte_lengths = []
        for tid in token_ids:
            raw_bytes = enc.decode_single_token_bytes(tid)
            byte_lengths.append(len(raw_bytes))
            # Decode to string; if multi-byte token is split, replace invalid bytes gracefully
            subwords.append(raw_bytes.decode("utf-8", errors="replace"))

        return {
            "encoding_name": encoding_name,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "subwords": subwords,
            "byte_lengths": byte_lengths,
            "error": None
        }
    except Exception as e:
        return {
            "encoding_name": encoding_name,
            "token_count": 0,
            "token_ids": [],
            "subwords": [],
            "byte_lengths": [],
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# 3. Hugging Face / AraBERT Analyzer
# ---------------------------------------------------------------------------
_HF_TOKENIZERS_CACHE: Dict[str, Any] = {}

def load_hf_tokenizer(model_name: str = "aubmindlab/bert-base-arabertv02"):
    """
    Loads and caches a Hugging Face tokenizer to avoid redundant downloads.
    """
    if model_name not in _HF_TOKENIZERS_CACHE:
        from transformers import AutoTokenizer
        # Load fast tokenizer if available
        _HF_TOKENIZERS_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _HF_TOKENIZERS_CACHE[model_name]


def analyze_with_hf(
    text: str,
    model_name: str = "aubmindlab/bert-base-arabertv02",
    include_special_tokens: bool = False
) -> Dict[str, Any]:
    """
    Analyzes input text using a Hugging Face tokenizer (e.g. AraBERT).
    
    Parameters:
        text: Input string to tokenize.
        model_name: Hugging Face model identifier (defaults to AraBERTv0.2).
        include_special_tokens: Whether to count [CLS] and [SEP] tokens.
                                Defaults to False for fair content comparison.
                                
    Returns:
        Dict containing:
            - model_name: Name of model/tokenizer
            - token_count: Total tokens
            - token_ids: Integer token IDs
            - subwords: WordPiece subword tokens (e.g. ['أ', '##ود'])
            - error: Error message if any
    """
    if not text.strip():
        return {
            "model_name": model_name,
            "token_count": 0,
            "token_ids": [],
            "subwords": [],
            "error": None
        }

    try:
        tokenizer = load_hf_tokenizer(model_name)
        token_ids = tokenizer.encode(text, add_special_tokens=include_special_tokens)
        subwords = tokenizer.tokenize(text)

        return {
            "model_name": model_name,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "subwords": subwords,
            "error": None
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "token_count": 0,
            "token_ids": [],
            "subwords": [],
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# 4. Metric Calculations: Arabic Token Inflation Factor
# ---------------------------------------------------------------------------
def calculate_inflation_factor(arabic_tokens: int, english_tokens: int) -> Optional[float]:
    """
    Calculates the Arabic Token Inflation Factor:
        Inflation Factor = Arabic Token Count / English Token Count
        
    Interpretation:
        - 1.0x: Equal number of tokens.
        - > 1.0x: Arabic uses more tokens (higher cost, faster context window fill).
        - < 1.0x: Arabic uses fewer tokens.
    """
    if english_tokens <= 0:
        return None
    return round(arabic_tokens / english_tokens, 2)


def get_inflation_interpretation(inflation_factor: Optional[float]) -> Dict[str, str]:
    """
    Returns a human-readable interpretation and UI styling badge.
    """
    if inflation_factor is None:
        return {
            "badge": "N/A",
            "description": "Cannot compute inflation factor (English token count is 0).",
            "color": "gray"
        }
    
    if inflation_factor > 1.05:
        pct = round((inflation_factor - 1.0) * 100, 1)
        return {
            "badge": f"+{pct}% Arabic Tokens",
            "description": f"Arabic uses {inflation_factor}x as many tokens as English ({pct}% higher).",
            "color": "orange" if inflation_factor < 2.0 else "red"
        }
    elif inflation_factor < 0.95:
        pct = round((1.0 - inflation_factor) * 100, 1)
        return {
            "badge": f"-{pct}% Arabic Tokens",
            "description": f"Arabic uses fewer tokens than English ({inflation_factor}x, {pct}% lower).",
            "color": "green"
        }
    else:
        return {
            "badge": "Approx. Equal (1.0x)",
            "description": "Arabic and English require approximately the same number of tokens.",
            "color": "blue"
        }


# ---------------------------------------------------------------------------
# 5. Full Comparative Analysis Pipeline
# ---------------------------------------------------------------------------
def run_full_comparison(
    english_text: str,
    arabic_text: str,
    hf_model_name: str = "aubmindlab/bert-base-arabertv02"
) -> Dict[str, Any]:
    """
    Executes a complete comparative tokenization pipeline on paired English & Arabic text.
    Computes metrics for both tiktoken (cl100k_base) and AraBERT.
    """
    # 1. Tiktoken Analysis
    tt_en = analyze_with_tiktoken(english_text, encoding_name="cl100k_base")
    tt_ar = analyze_with_tiktoken(arabic_text, encoding_name="cl100k_base")
    tt_inflation = calculate_inflation_factor(tt_ar["token_count"], tt_en["token_count"])

    # 2. Hugging Face / AraBERT Analysis
    hf_en = analyze_with_hf(english_text, model_name=hf_model_name, include_special_tokens=False)
    hf_ar = analyze_with_hf(arabic_text, model_name=hf_model_name, include_special_tokens=False)
    hf_inflation = calculate_inflation_factor(hf_ar["token_count"], hf_en["token_count"])

    return {
        "inputs": {
            "english": english_text,
            "arabic": arabic_text
        },
        "tiktoken": {
            "english": tt_en,
            "arabic": tt_ar,
            "inflation_factor": tt_inflation,
            "interpretation": get_inflation_interpretation(tt_inflation)
        },
        "huggingface": {
            "english": hf_en,
            "arabic": hf_ar,
            "inflation_factor": hf_inflation,
            "interpretation": get_inflation_interpretation(hf_inflation)
        }
    }


# ---------------------------------------------------------------------------
# 6. Benchmark Across Built-in Samples
# ---------------------------------------------------------------------------
def benchmark_built_in_samples(
    hf_model_name: str = "aubmindlab/bert-base-arabertv02"
) -> pd.DataFrame:
    """
    Runs dynamic tokenization analysis across all built-in Banking and News samples.
    Generates a structured DataFrame suitable for summary tables and charts.
    """
    rows = []
    
    for category, examples in SAMPLE_DATASETS.items():
        for name, pair in examples.items():
            en_text = pair["english"]
            ar_text = pair["arabic"]
            
            # Tiktoken
            tt_en = analyze_with_tiktoken(en_text, "cl100k_base")
            tt_ar = analyze_with_tiktoken(ar_text, "cl100k_base")
            tt_inf = calculate_inflation_factor(tt_ar["token_count"], tt_en["token_count"])
            
            rows.append({
                "Domain": category,
                "Example": name,
                "Tokenizer": "GPT-style (tiktoken cl100k_base)",
                "English Tokens": tt_en["token_count"],
                "Arabic Tokens": tt_ar["token_count"],
                "Inflation Factor": f"{tt_inf:.2f}x" if tt_inf else "N/A",
                "Raw Inflation": tt_inf or 1.0
            })
            
            # Hugging Face / AraBERT
            hf_en = analyze_with_hf(en_text, hf_model_name)
            hf_ar = analyze_with_hf(ar_text, hf_model_name)
            hf_inf = calculate_inflation_factor(hf_ar["token_count"], hf_en["token_count"])
            
            rows.append({
                "Domain": category,
                "Example": name,
                "Tokenizer": f"AraBERT ({hf_model_name.split('/')[-1]})",
                "English Tokens": hf_en["token_count"],
                "Arabic Tokens": hf_ar["token_count"],
                "Inflation Factor": f"{hf_inf:.2f}x" if hf_inf else "N/A",
                "Raw Inflation": hf_inf or 1.0
            })

    return pd.DataFrame(rows)
