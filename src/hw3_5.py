import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma
from collections import Counter

np.random.seed(42)


# Load the tokens from the data file, treating periods as their own token
with open('data/alice.txt', 'r') as f:
    lines = f.readlines()
    tokens = []
    for line in lines:
        for word in line.split():
            if word.endswith("."):
                word = word[:-1]
                tokens.append(word)
                tokens.append(".")
            else:
                tokens.append(word)


# Create a vocabulary
vocab = sorted(set(tokens))
w2i = {w: idx for idx, w in enumerate(vocab)}
i2w = {idx: w for w, idx in w2i.items()}
k = len(vocab)
n = len(tokens)
print(f"Tokens n = {n},  Vocabulary size k = {k}\n")
# print("Vocabulary:", vocab)


## i.i.d. model
prior_alpha = 1e-3
counts = Counter(tokens)
alpha = np.array([counts[w] + prior_alpha for w in vocab], dtype=float)

words_by_alpha = sorted(vocab, key=lambda w: alpha[w2i[w]], reverse=True)
# print("Top 5 words by alpha:")
# print(words_by_alpha[:5], "\n")

M = 20
for s in range(M):
    p = np.random.dirichlet(alpha)    
    sentence = []
    while True:
        word_idx = np.random.choice(k, p=p)
        word = i2w[word_idx]
        sentence.append(word)
        if word == '.':
            break
    # print(f"  Sentence {s+1}: {' '.join(sentence)}")
    
    
## AR model
def log_evidence(a):
    A = a.sum()
    val = 0.0
    for i in range(k):
        if x_prev[i] == 0:
            continue
        val += gammaln(A) - gammaln(x_prev[i] + A)
        val += (gammaln(X[:, i] + a) - gammaln(a)).sum()
    return val

def grad_log_evidence(a):
    A = a.sum()
    g = np.zeros(k)
    for i in range(k):
        if x_prev[i] == 0:
            continue
        # vector part: sum_i [ ψ(x_{j|i}+a_j) - ψ(a_j) ]
        g += digamma(X[:, i] + a) - digamma(a)
        # scalar part from A: sum_i [ ψ(A) - ψ(x_i + A) ] added to every component
        g += digamma(A) - digamma(x_prev[i] + A)
    return g

X = np.zeros((k, k), dtype=np.int64)
for t in range(1, n):
    i_prev = w2i[tokens[t - 1]]  # i = previous word
    j_next = w2i[tokens[t]]      # j = next word
    X[j_next, i_prev] += 1

# x_i = sum_j x_{j|i}  (times i appears as the previous word)
x_prev = X.sum(axis=0)  # shape (k,)
a = 1 * np.ones(k) #initial values of a_1, \dots, a_k

print("Optimizing hyperparameters a = (a_1,...,a_k)...")
for it in range(3000):
    log_a = np.log(a)
    le = log_evidence(a)
    g = grad_log_evidence(a) * a  # chain rule: d/d(log a) = a * d/da
    step = 0.01
    for _ in range(10):
        trial = np.exp(log_a + step * g).clip(1e-10)
        if log_evidence(trial) > le:
            break
        step *= 0.5
    a = np.exp(log_a + step * g).clip(1e-10)
    if it % 100 == 0 or it == 299:
        print(f"  iter {it:3d}: log_ev = {le:.2f}, A = {a.sum():.3f}")

A = a.sum()
print(f"\nOptimal A = {A:.3f}\n")
# print(f"{'word':>8s}  {'a_j':>10s}")
# print("-" * 30)
# for j in range(k):
#     print(f"{vocab[j]:>8s}  {a[j]:10.4f}")
    
    
# Calculate posterior mean estimates for each bigram
P_hat = np.zeros((k, k))
for i in range(k):
    P_hat[:, i] = (X[:, i] + a) / (x_prev[i] + A)
print(P_hat)

# Report the top-k most probable next words for each context work
def predict_next(prev_word, top_k=5):
    i = w2i[prev_word]
    probs = P_hat[:, i]                 # probs over j given i
    ranked = np.argsort(probs)[::-1]
    return [(i2w[j], probs[j]) for j in ranked[:top_k]]

for ctx in ['alice', 'she', 'the']:
    i = w2i[ctx]
    lam = A / (x_prev[i] + A)  # shrinkage weight on the prior mean m
    print(f"\nAfter '{ctx}'  (x_i={int(x_prev[i])}, λ={lam:.3f}):")
    for word, prob in predict_next(ctx, top_k=4):
        bar = '█' * int(prob * 40)
        print(f"  {word:>8s}  {prob:.3f}  {bar}")


# Generate sentences with the AR model
def generate(start_word, length=12, seed=45):
    rng = np.random.default_rng(seed)
    words = [start_word]
    for _ in range(length):
        i = w2i[words[-1]]
        j = rng.choice(k, p=P_hat[:, i])
        words.append(i2w[j])
        if words[-1] == '.':
            break
    return ' '.join(words)

for s in range(20):
    print("s: " + generate('.', seed=45 + s))
