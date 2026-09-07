"""
Automated Verification Script for Tokenizer Analysis
"""
import sys

def test_imports():
    print("Testing package imports...")
    import streamlit
    import tiktoken
    import transformers
    import tokenizers
    import pandas
    import altair
    print(f"  [OK] streamlit {streamlit.__version__}")
    print(f"  [OK] tiktoken {tiktoken.__version__}")
    print(f"  [OK] transformers {transformers.__version__}")
    print(f"  [OK] pandas {pandas.__version__}")

def test_tokenizer_analysis():
    print("\nTesting tokenizer_analysis logic...")
    from tokenizer_analysis import (
        analyze_with_tiktoken,
        analyze_with_hf,
        calculate_inflation_factor,
        get_inflation_interpretation,
        benchmark_built_in_samples,
        SAMPLE_DATASETS
    )

    en_text = "I would like to open a new savings account with your bank today."
    ar_text = "أود فتح حساب توفير جديد لدى بنككم اليوم."

    # 1. Tiktoken
    res_tt_en = analyze_with_tiktoken(en_text, "cl100k_base")
    res_tt_ar = analyze_with_tiktoken(ar_text, "cl100k_base")
    assert res_tt_en["token_count"] > 0, "English tiktoken count should be > 0"
    assert res_tt_ar["token_count"] > 0, "Arabic tiktoken count should be > 0"
    assert len(res_tt_en["subwords"]) == res_tt_en["token_count"]
    print(f"  [OK] Tiktoken EN tokens: {res_tt_en['token_count']}, AR tokens: {res_tt_ar['token_count']}")

    inf_tt = calculate_inflation_factor(res_tt_ar["token_count"], res_tt_en["token_count"])
    assert inf_tt is not None
    print(f"  [OK] Tiktoken Arabic Inflation Factor: {inf_tt}x")

    # 2. Hugging Face AraBERT
    print("  Loading AraBERT tokenizer...")
    res_hf_en = analyze_with_hf(en_text, "aubmindlab/bert-base-arabertv02")
    res_hf_ar = analyze_with_hf(ar_text, "aubmindlab/bert-base-arabertv02")
    assert res_hf_en["token_count"] > 0
    assert res_hf_ar["token_count"] > 0
    print(f"  [OK] AraBERT EN tokens: {res_hf_en['token_count']}, AR tokens: {res_hf_ar['token_count']}")

    inf_hf = calculate_inflation_factor(res_hf_ar["token_count"], res_hf_en["token_count"])
    print(f"  [OK] AraBERT Arabic Inflation Factor: {inf_hf}x")

    # 3. Benchmark
    print("  Running built-in benchmark...")
    df = benchmark_built_in_samples("aubmindlab/bert-base-arabertv02")
    assert len(df) > 0
    print(f"  [OK] Benchmark generated {len(df)} rows across Banking & News domains.")
    print("\nAll verification checks PASSED successfully!")

if __name__ == "__main__":
    test_imports()
    test_tokenizer_analysis()
