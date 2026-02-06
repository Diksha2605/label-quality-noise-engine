def summarize_results(baseline_acc, clean_acc):
    return {
        "baseline_accuracy": round(baseline_acc, 3),
        "cleaned_accuracy": round(clean_acc, 3),
        "improvement": round(clean_acc - baseline_acc, 3),
        "conclusion": "Cleaning low-trust samples improves generalization"
        if clean_acc > baseline_acc
        else "No improvement observed",
    }
