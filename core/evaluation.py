from collections import Counter


# -------------------------
# DOCUMENT CONTRIBUTION ANALYSIS
# -------------------------
def analyze_document_contribution(retrieved_docs):

    source_counter = Counter()

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        source_counter[source] += 1

    total = sum(source_counter.values())

    breakdown = []

    for source, count in source_counter.items():
        percentage = round((count / total) * 100, 2)
        breakdown.append({
            "source": source,
            "chunks_used": count,
            "contribution_percent": percentage
        })

    return breakdown


# -------------------------
# RETRIEVAL CONFIDENCE SCORE
# -------------------------
def calculate_retrieval_confidence(retrieved_docs):

    if not retrieved_docs:
        return 0

    score = len(retrieved_docs)

    # Basic heuristic scoring
    if score >= 8:
        return 90
    elif score >= 5:
        return 75
    elif score >= 3:
        return 60
    else:
        return 40
