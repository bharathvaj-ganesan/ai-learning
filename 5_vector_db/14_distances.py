"""
VECTOR DB 1 of 4 - DISTANCE METRICS (3 min)

A vector DB answers one question: "which stored vectors are closest to this one?"
"Closest" needs a definition. Three common ones:

  dot product   a . b           bigger = more similar, depends on length
  cosine        a.b / |a||b|    angle only, ignores length   (text embeddings)
  L2 distance   |a - b|         straight-line distance, smaller = more similar

Trick used by every index in this folder: normalize stored vectors to length 1
at insert time, and normalize the query at search time. Normalizing means
dividing by the vector's length: it changes length, but preserves direction.
Then cosine == dot product, and scoring all vectors is one matrix multiply.
Normalization requires nonzero vectors: a zero vector has no direction.

Run:  python 5_vector_db/14_distances.py
Next: python 5_vector_db/15_flat_search.py
"""
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])   # same direction as a, twice as long
c = np.array([3.0, 0.0, -1.0])  # different direction

def dot(x, y):    return float(x @ y)
def cosine(x, y): return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
def l2(x, y):     return float(np.linalg.norm(x - y))

print(f"{'pair':<8}{'dot':>8}{'cosine':>9}{'L2':>8}")

# Each dictionary entry gives a display label and the two vectors to compare.
# .items() yields (key, value); name gets the key, and (x, y) unpacks the value.
# For example, the first iteration sets name="a,b", x=a, and y=b.
for name, (x, y) in {"a,b": (a, b), "a,c": (a, c)}.items():
    # Formatting only: <8 left-aligns in 8 characters; >8.2f right-aligns
    # a number in 8 characters with 2 decimals. >9.3f uses 9 and 3 instead.
    print(f"{name:<8}{dot(x, y):>8.2f}{cosine(x, y):>9.3f}{l2(x, y):>8.2f}")

print("\nReading the table:")
print("  a and b: b = 2*a, so they point in exactly the same direction.")
print("    Dot = 1*2 + 2*4 + 3*6 = 28; longer vectors can have larger dot products.")
print("    Cosine = 1.0 means the same direction, regardless of length.")
print("    L2 = sqrt((1-2)^2 + (2-4)^2 + (3-6)^2) = sqrt(14), about 3.74.")
print("    L2 measures how far apart the endpoints are, so scaling changes it.")
print("  a and c: their dot product is 1*3 + 2*0 + 3*(-1) = 0.")
print("    Their cosine is also 0: they are perpendicular (a 90-degree angle).")

# Now search 1,000 example vectors, each with 32 coordinates.
# A fixed seed makes the random example repeatable.
rng = np.random.default_rng(0)
V = rng.normal(size=(1000, 32))
# axis=1 computes one length per row. keepdims=True gives shape (1000, 1),
# so NumPy divides every coordinate in each row by that row's own length.
V /= np.linalg.norm(V, axis=1, keepdims=True)
q = rng.normal(size=32)  # the query: the vector we want to find neighbors for
q /= np.linalg.norm(q)

# V @ q returns 1,000 dot products, one similarity score per stored vector.
# Since both sides have length 1, these are also their cosine similarities.
dot_scores = V @ q
# V - q subtracts the query from every row; axis=1 gives one distance per row.
l2_distances = np.linalg.norm(V - q, axis=1)

# argsort returns row indices ordered by score, not the score values themselves.
# It sorts ascending: negate dot scores to put the largest similarities first.
# L2 already wants the smallest distances first. [:5] keeps five neighbors.
by_dot = np.argsort(-dot_scores)[:5]
by_l2 = np.argsort(l2_distances)[:5]
print("\nTop-5 by dot :", by_dot)
print("Top-5 by L2  :", by_l2)
print("\nWhy the rankings match after normalization:")
print("  |v-q|^2 = |v|^2 + |q|^2 - 2*(v.q) = 2 - 2*(v.q).")
print("  As the dot product increases, squared L2 decreases.")
print("  Taking a square root preserves distance order, so the neighbors match.")
print("  The scores differ: dot/cosine = 1 means identical; L2 = 0 means identical.")
print("  Equal scores can tie, and floating-point rounding can affect near-ties.")
