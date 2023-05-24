from collections import Counter
import math


def count_ngrams(sentence, n):
    """
    Count the n-grams in the given sentence.

    Args:
    sentence (list): A list of words/tokens in the sentence.
    n (int): The "n" in n-gram, e.g. 1 for unigram, 2 for bigram, etc.

    Returns:
    Counter: A Counter object with each n-gram as the key and its count as the value.
    """
    # Use a list comprehension to generate n-grams and count them using Counter
    return Counter([tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)])


def sentence_bleu(references, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    Calculate the sentence-level BLEU score for a candidate sentence.

    Args:
    references (list of list): A list of reference translations, each as a list of words/tokens.
    candidate (list): The candidate translation as a list of words/tokens.
    weights (list): The weights for the BLEU score calculation, typically [0.25, 0.25, 0.25, 0.25].

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
        clipped_counts = dict((ngram, min(count, candidate_counts[ngram])) for ngram, count in reference_counts.items())
        p_ns.append(sum(clipped_counts.values()) / float(max(sum(candidate_counts.values()), 1)))

    # Calculate final score
    score = bp * math.exp(sum(w_i*math.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_ns)))

    return score


if __name__ == "__main__":

    # References
    references = [
        ['the', 'cat', 'is', 'on', 'the', 'mat']
    ]

    # Candidate (machine-generated) sentence
    candidate = ['the', 'cat', 'is', 'on', 'mat']

    # Calculate and print the BLEU score
    score = sentence_bleu(references, candidate)
    print(score)
