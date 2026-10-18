def ctc_decode(indices, blank=0):
    result, previous = [], None
    for token in indices:
        token = int(token)
        if token != previous and token != blank:
            result.append(token)
        previous = token
    return result


def ctc_minimum_length(target):
    return len(target)+sum(a == b for a, b in zip(target[:-1], target[1:]))


def edit_counts(reference, hypothesis):
    """Levenshtein path with deterministic tie ordering substitution/deletion/insertion."""
    n, m = len(reference), len(hypothesis)
    table = [[(0, 0, 0)]*(m+1) for _ in range(n+1)]
    for i in range(n+1): table[i][0] = (0, i, 0)
    for j in range(m+1): table[0][j] = (0, 0, j)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if reference[i-1] == hypothesis[j-1]:
                table[i][j] = table[i-1][j-1]
            else:
                s, d, ins = table[i-1][j-1]
                substitution = (s+1, d, ins)
                s, d, ins = table[i-1][j]
                deletion = (s, d+1, ins)
                s, d, ins = table[i][j-1]
                insertion = (s, d, ins+1)
                table[i][j] = min([substitution, deletion, insertion], key=sum)
    return table[n][m]


def word_error_rate(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError('Reference/hypothesis count mismatch')
    counts = [edit_counts(r, h) for r, h in zip(references, hypotheses)]
    s, d, i = (sum(c[k] for c in counts) for k in range(3))
    n = sum(map(len, references))
    return {'WER': (s+d+i)/n if n else None, 'substitutions': s, 'deletions': d, 'insertions': i, 'reference_tokens': n,
            'num_utterances': len(references)}
