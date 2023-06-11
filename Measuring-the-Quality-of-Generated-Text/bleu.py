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
    max_ref_cnt_dict = {}

    # For each reference sentence
    for ref in reference_list: 
        # Count n-grams in the reference sentence
        ref_cnt = simple_count(ref, n)

        # For each n-gram in the reference sentence, update its maximum count
        for n_gram in ref_cnt:
            max_ref_cnt_dict[n_gram] = max(ref_cnt[n_gram], max_ref_cnt_dict.get(n_gram, 0))

    # Compute clipped counts
    return {
        # The clipped count of each n-gram is the minimum of its count in the candidate sentence
        # and its maximum count in any of the reference sentences.
        n_gram: min(ca_cnt.get(n_gram, 0), max_ref_cnt_dict.get(n_gram, 0)) for n_gram in ca_cnt
    }


def modified_precision(candidate, reference_list, n):
    """
    Calculate the modified precision of a candidate translation text.
    Modified precision is a part of BLEU (Bilingual Evaluation Understudy) score calculation. 
    It measures how frequently the predicted n-grams appear in the reference text.
    
    Args:
    candidate (list): The candidate translation text as a list of words.
    reference_list (list): The reference translation texts as a list of words.
    n (int): The size of the n-gram.
    
    Returns:
    float: The modified precision score.
    """
    # Count the maximum number of times that each n-gram occurs in any single reference translation
    clip_cnt = count_clip(candidate, reference_list, n)

    # Calculate the sum of clipped counts for the numerator of the modified precision
    total_clip_cnt = sum(clip_cnt.values())

    # Count the number of n-grams in the candidate translation
    cnt = simple_count(candidate, n)

    # Calculate the sum of counts for the denominator of the modified precision
    total_cnt = sum(cnt.values())

    # To avoid ZeroDivisionError if total count is 0
    total_cnt = 1 if total_cnt == 0 else total_cnt

    # Return the modified precision as the ratio of total clipped count to total count
    return (total_clip_cnt / total_cnt)


def closest_ref_length(candidate, reference_list):
    """
    Given a candidate string, find and return the length of the reference string that is closest in length to the candidate.

    This function takes a candidate string and a list of reference strings as input, and returns the length of the reference string whose length is closest to the candidate string's length. 
    If there are multiple reference strings with the same length difference, it chooses the one with the shorter length.
    """
    ca_len = len(candidate)
    ref_lens = (len(ref) for ref in reference_list)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - ca_len), ref_len))
    return closest_ref_len


def brevity_penalty(candidate, reference_list):
    """Computes the brevity penalty for BLEU (Bilingual Evaluation Understudy) score calculation."""

    ca_len = len(candidate)
    if ca_len == 0:
        return 0

    ref_len = closest_ref_length(candidate, reference_list)
    return min(1.0, np.exp(1 - ref_len / ca_len))


def bleu_score(candidate, reference_list, weights=[0.25, 0.25, 0.25, 0.25]):
    """Computes the BLEU (Bilingual Evaluation Understudy) score for a candidate string."""

    # Compute the brevity penalty.
    bp = brevity_penalty(candidate, reference_list)

    # Compute the modified precision for each order of n-grams.
    p_n = [modified_precision(candidate, reference_list, n=n) for n, _ in enumerate(weights,start=1)] 

    # Compute the weighted average of the log precisions, using the provided weights.
    # score = np.sum([w_i * np.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_n)])
    score = np.sum([w_i * np.log(p_i + 1e-15) for w_i, p_i in zip(weights, p_n)])

    # Return the BLEU score, which is the brevity penalty times the exponential of the average log precision.
    return bp * np.exp(score)


if __name__ == '__main__':
    import nltk.translate.bleu_score as bleu

    candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    references = [
        'It is a guide to action that ensures that the military will forever heed Party commands',
        'It is the guiding principle which guarantees the military forces always being under the command of the Party',
        'It is the practical guide for the army always to heed the directions of the party'
    ]

    print('실습 코드의 BLEU :',bleu_score(candidate.split(),list(map(lambda ref: ref.split(), references))))
    print('패키지 NLTK의 BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate.split()))
    # 실습 코드의 BLEU : 0.5045666840058485
    # 패키지 NLTK의 BLEU : 0.5045666840058485
