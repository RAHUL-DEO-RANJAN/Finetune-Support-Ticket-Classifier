# Test script to verify all imports work
print("Testing imports...")

try:
    import pandas
    print("[OK] pandas")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import sklearn
    print("[OK] scikit-learn")
except ImportError as e:
    print(f"[FAIL] scikit-learn: {e}")

try:
    import streamlit
    print("[OK] streamlit")
except ImportError as e:
    print(f"[FAIL] streamlit: {e}")

try:
    import matplotlib
    print("[OK] matplotlib")
except ImportError as e:
    print(f"[FAIL] matplotlib: {e}")

try:
    import joblib
    print("[OK] joblib")
except ImportError as e:
    print(f"[FAIL] joblib: {e}")

try:
    import transformers
    print("[OK] transformers")
except ImportError as e:
    print(f"[FAIL] transformers: {e}")

try:
    import torch
    print("[OK] torch")
except ImportError as e:
    print(f"[FAIL] torch: {e}")

try:
    import datasets
    print("[OK] datasets")
except ImportError as e:
    print(f"[FAIL] datasets: {e}")

try:
    import accelerate
    print("[OK] accelerate")
except ImportError as e:
    print(f"[FAIL] accelerate: {e}")

try:
    import evaluate
    print("[OK] evaluate")
except ImportError as e:
    print(f"[FAIL] evaluate: {e}")

print("\nAll imports checked!")
