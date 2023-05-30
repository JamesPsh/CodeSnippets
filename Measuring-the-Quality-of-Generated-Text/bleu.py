'''
BLEU Score(Bilingual Evaluation Understudy Score)
https://wikidocs.net/31695
'''

import numpy as np
from collections import Counter
from nltk import ngrams



# 토큰화 된 문장(tokens)에서 n-gram을 카운트
def simple_count(tokens, n):
    """
    This function counts the occurrences of n-grams in the given list of tokens.

    Args:
        tokens (list): A list of tokens from a sentence.
        n (int): The size of the n-gram. For example, use 1 for unigram, 2 for bigram, etc.

    Returns:
        Counter: A Counter dictionary that maps each n-gram to its frequency count.

    Example:
        >>> simple_count(['I', 'am', 'studying', 'NLP'], 2)
        Counter({('I', 'am'): 1, ('am', 'studying'): 1, ('studying', 'NLP'): 1})

    """
    # Use nltk's ngrams function to generate n-grams
    # Then use Counter to count the frequency of each n-gram
    return Counter(ngrams(tokens, n))


def count_clip(candidate, reference_list, n):
    """
    This function calculates the clipped count of n-grams in the candidate sentence.

    The clipped count of each n-gram is the minimum of its count in the candidate sentence
    and its maximum count in any of the reference sentences.

    Args:
        candidate (list): A list of tokens from the candidate sentence.
        reference_list (list of list): A list of lists of tokens from the reference sentences.
        n (int): The size of the n-gram. For example, use 1 for unigram, 2 for bigram, etc.

    Returns:
        dict: A dictionary that maps each n-gram in the candidate sentence to its clipped count.

    Example:
        >>> count_clip(['I', 'am', 'studying', 'NLP'], [['I', 'am', 'learning', 'NLP']], 2)
        {('I', 'am'): 1, ('am', 'studying'): 0, ('studying', 'NLP'): 1}

    """
    # Count n-grams in the candidate sentence
    ca_cnt = simple_count(candidate, n)
    max_ref_cnt_dict = Counter()

    # For each reference sentence
    for ref in reference_list: 
        # Count n-grams in the reference sentence
        ref_cnt = simple_count(ref, n)

        # For each n-gram in the reference sentence, update its maximum count
        for n_gram in ref_cnt:
            max_ref_cnt_dict[n_gram] = max(ref_cnt[n_gram], max_ref_cnt_dict[n_gram])

    # Compute clipped counts
    return {
        # The clipped count of each n-gram is the minimum of its count in the candidate sentence
        # and its maximum count in any of the reference sentences.
        n_gram: min(ca_cnt.get(n_gram, 0), max_ref_cnt_dict.get(n_gram, 0)) for n_gram in ca_cnt
    }
