"""
VECTOR DB 1 of 4 - DISTANCE METRICS

A vector DB compares a query vector with stored vectors.

  cosine: compare direction; bigger is better
  L2:     compare physical distance; smaller is better

Text embeddings usually use direction. We normalize every vector to length 1,
which makes dot product equal cosine similarity.

Run:  python 5_vector_db/14_distances.py
Next: python 5_vector_db/15_flat_search.py
"""
import numpy as np


# ---------------------------------------------------------------------------
# BLOCK 1: Make every arrow the same length
# ---------------------------------------------------------------------------
def normalize(v):
    """Shorten or stretch an arrow to length 1 without turning it."""
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# BLOCK 2: Create a tiny vector database
# ---------------------------------------------------------------------------
def make_example():
    """Return one search arrow and four stored arrows."""
    query = normalize(np.array([1.0, 0.0]))  # points right

    stored = {
        "same direction": normalize(np.array([2.0, 0.0])),
        "slightly upward": normalize(np.array([2.0, 1.0])),
        "straight upward": normalize(np.array([0.0, 1.0])),
        "opposite": normalize(np.array([-1.0, 0.0])),
    }
    return query, stored


# ---------------------------------------------------------------------------
# BLOCK 3: Compare one stored arrow with the query
# ---------------------------------------------------------------------------
def compare(vector, query):
    """Return angle similarity and straight-line distance."""
    cosine = float(vector @ query)              # bigger is better
    l2 = float(np.linalg.norm(vector - query))  # smaller is better
    return cosine, l2


# ---------------------------------------------------------------------------
# BLOCK 4: Search all stored arrows
# ---------------------------------------------------------------------------
def search(stored, query):
    """Compare everything, then put the best matches first."""
    results = []
    for name, vector in stored.items():
        cosine, l2 = compare(vector, query)
        results.append((name, cosine, l2))

    return sorted(results, key=lambda row: row[1], reverse=True)


# ---------------------------------------------------------------------------
# BLOCK 5: Show the lesson
# ---------------------------------------------------------------------------
def main():
    query, stored = make_example()
    results = search(stored, query)

    print("QUESTION")
    print("  Which stored arrow points most like the query arrow?")
    print("  Query: [1, 0], an arrow pointing right.\n")

    print("SCORES")
    print(f"  {'stored arrow':<20} {'cosine':>8} {'L2':>8}")
    print("  " + "-" * 38)
    for name, cosine, l2 in results:
        print(f"  {name:<20} {cosine:>8.3f} {l2:>8.3f}")

    print("\nHOW TO READ THE SCORES")
    print("  Cosine: 1 = same direction, 0 = sideways, -1 = opposite.")
    print("  L2:     0 = same place; a larger number means farther away.")

    print("\nSEARCH RESULT — BEST MATCH FIRST")
    for rank, (name, cosine, l2) in enumerate(results, start=1):
        print(f"  {rank}. {name:<20} cosine={cosine:.3f}, L2={l2:.3f}")

    print("\nTAKEAWAY")
    print("  All arrows were first made length 1.")
    print("  Therefore, highest cosine and lowest L2 give the same order.")
    print("  Asking for top-2 would return the first two rows above.")


if __name__ == "__main__":
    main()
