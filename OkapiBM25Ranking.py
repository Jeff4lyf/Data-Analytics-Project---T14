import math
from collections import Counter

def bm25_scores(corpus, query_terms, k1=1.5, b=0.75):
    docs = [doc.lower().split() for doc in corpus]
    num_docs = len(docs)
    doc_lengths = [len(doc) for doc in docs]
    avgdl = sum(doc_lengths) / num_docs if num_docs > 0 else 0
    
    query_tokens = [term.lower() for term in query_terms]
    
    term_doc_freqs = Counter()
    for doc in docs:
        unique_terms = set(doc)
        for term in query_tokens:
            if term in unique_terms:
                term_doc_freqs[term] += 1
                
    print(f"Corpus Stats:")
    print(f"  N: {num_docs}")
    print(f"  avgdl: {avgdl:.2f}")
    print(f"  Query Term Docs (n(q)): {dict(term_doc_freqs)}")
    print("-" * 20)
    
    idfs = {}
    for term in query_tokens:
        df = term_doc_freqs[term]
        numerator = num_docs - df + 0.5
        denominator = df + 0.5
        if denominator == 0:
            idf = 0.0
        else:
            idf = math.log(numerator / denominator)
        idfs[term] = idf
        
    print(f"Query Term IDFs: { {k: f'{v:.2f}' for k, v in idfs.items()} }")
    print("-" * 20)
    
    scores = []
    for i, doc in enumerate(docs):
        term_freqs = Counter(doc)
        doc_len = doc_lengths[i]
        
        doc_score = 0
        for term in query_tokens:
            if term in term_freqs:
                f = term_freqs[term]
                idf = idfs[term]
                numerator_term = f * (k1 + 1)
                denominator_term = f + k1 * (1 - b + b * (doc_len / avgdl))
                
                term_score = idf * (numerator_term / denominator_term)
                doc_score += term_score
        
        scores.append(doc_score)
        
    return scores, idfs

corpus_raw = [
    "cat mats on mats",
    "dog runs fast",
    "cat dog play together"
]
query_terms_raw = ["cat", "mats"]

print(f"Corpus:\n" + "\n".join(corpus_raw) + "\n")
print(f"Query: {query_terms_raw}\n")

scores, idfs = bm25_scores(corpus_raw, query_terms_raw)

ranked_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

print("\nBM25 Ranking and Scores:")
for rank, idx in enumerate(ranked_indices):
    print(f"{rank+1}. Document D{idx+1} (Score: {scores[idx]:.2f}): \"{corpus_raw[idx]}\"")
