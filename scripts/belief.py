import numpy as np
import scipy

CLASS_COUNT = 5

def prob(dist):
    probs = np.exp(dist - np.max(dist))
    probs /= probs.sum()
    return probs

def output(dist):
    probs = prob(dist)
    for pred, p in enumerate(probs): print(f"{pred}: {100 * p:.2f}%")

def make_distribution(pred, conf):
    other_prob = (1.0 - conf) / (CLASS_COUNT - 1)
    probs = np.full(CLASS_COUNT, other_prob)
    probs[pred] = conf
    return np.log(probs)

def update(dist, pred, conf):
    dist[:] += make_distribution(pred, conf)

def js_agreement(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    js_dist = np.sqrt(js)
    agreement = 1.0 - js_dist
    return agreement

def weight(dist, pred, conf):
    query = prob(make_distribution(pred, conf))
    belief = prob(dist)
    print(f'dot: {np.dot(query, belief)}')
    print(f'kl: {scipy.special.kl_div(query, belief)}')
    print(f'js: {js_agreement(query, belief)}')

belief = make_distribution(0, 1.0 / CLASS_COUNT)

output(belief)

while True:
    print('updating belief...')
    try:
        pred = int(input("prediction: "))
        conf = float(input("confidence: "))
        n = int(input("times: "))

        for i in range(0, n):
            update(belief, pred, conf)

        output(belief)

    except KeyboardInterrupt:
        print()
        break
    except ValueError:
        print("invalid input...")

output(belief)

while True:
    print('querying weight...')
    try:
        pred = int(input("prediction: "))
        conf = float(input("confidence: "))

        weight(belief, pred, conf)

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except ValueError:
        print("invalid input...")