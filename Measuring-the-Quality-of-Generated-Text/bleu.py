from collections import Counter
import math
from typing import List, Tuple


def count_ngrams(sentence: List[str], n: int) -> Counter:
    """
    Count the n-grams in the given sentence.

    Args:
        sentence (List[str]): A list of words/tokens in the sentence.
        n (int): The "n" in n-gram, e.g. 1 for unigram, 2 for bigram, etc.

    Returns:
        Counter: A Counter object with each n-gram as the key and its count as the value.
    """
    ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
    return Counter(ngrams)


def sentence_bleu(references: List[List[str]], candidate: List[str], weights: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """
    Calculate the sentence-level BLEU score for a candidate sentence.

    Args:
        references (List[List[str]]): A list of reference translations, each as a list of words/tokens.
        candidate (List[str]): The candidate translation as a list of words/tokens.
        weights (List[float]): The weights for the BLEU score calculation, typically [0.25, 0.25, 0.25, 0.25].

    Returns:
        float: The BLEU score.
    """

    # Brevity Penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - len(candidate)), ref_len))
    if len(candidate) < closest_ref_len:
        bp = math.exp(1 - float(closest_ref_len) / len(candidate))
    else:
        bp = 1.0

    # Modified precision for each gram
    p_ns = []  
    for k in range(1, 5):
        candidate_counts = count_ngrams(candidate, k)
        reference_counts = Counter()

        for reference in references:
            reference_counts = reference_counts | count_ngrams(reference, k)

    # Calculate final score
        if not candidate_counts:
            p_ns.append(0)
        else:
            clipped_counts = {ngram: min(count, candidate_counts[ngram]) for ngram, count in reference_counts.items()}
            p_ns.append(sum(clipped_counts.values()) / float(max(sum(candidate_counts.values()), 1)))

    score = bp * math.exp(sum(w_i*math.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_ns)))

    return score


if __name__ == "__main__":
    references = [['the', 'cat', 'is', 'on', 'the', 'mat']]
    candidate = ['the', 'cat', 'is', 'on', 'mat']
    score = sentence_bleu(references, candidate)
    print(score)
