from jiwer import wer

def calculate_wer(predictions, references):
    total_wer = 0
    for pred, ref in zip(predictions, references):
        total_wer += wer(ref, pred)
    avg_wer = total_wer / len(references)
    print(f"Word Error Rate: {avg_wer:.2f}")
    return avg_wer
