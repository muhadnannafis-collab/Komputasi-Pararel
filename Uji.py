import os, re, time, random, json
from collections import Counter
from multiprocessing import Pool

DATA_DIR = "corpus"
TOKEN = re.compile(r"[a-z]+")

def generate_corpus(n_docs=240, words_per_doc=60000, vocab=5000, seed=42):
    """Membuat dokumen sintetis berdistribusi Zipf."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rnd = random.Random(seed)
    words = ["".join(rnd.choices("abcdefghijklmnopqrstuvwxyz", k=rnd.randint(3, 9))) for _ in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]
    for i in range(n_docs):
        with open(f"{DATA_DIR}/doc_{i:04d}.txt", "w") as f:
            f.write(" ".join(rnd.choices(words, weights, k=words_per_doc)))

def count_file(path):                      
    with open(path, encoding="utf-8") as f:
        return Counter(TOKEN.findall(f.read().lower()))

def sequential(files):
    total = Counter()
    for p in files:
        total.update(count_file(p))
    return total

def parallel(files, workers):
    total = Counter()
    with Pool(workers) as pool:            
        for c in pool.imap_unordered(count_file, files, chunksize=4):
            total.update(c)                
    return total

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        generate_corpus()
    files = sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR))
    size_mb = sum(os.path.getsize(p) for p in files) / 1e6
    print(f"CPU: {os.cpu_count()} | dokumen: {len(files)} | ukuran: {size_mb:.1f} MB")

    t = time.perf_counter(); ref = sequential(files); t_seq = time.perf_counter() - t
    print(f"Sekuensial: {t_seq:.2f} s | kata unik: {len(ref)}")
    res = {"seq": t_seq, "docs": len(files), "mb": size_mb, "unique": len(ref),
           "top": ref.most_common(5), "par": {}}
    for w in [1, 2, 4, 8]:
        if w > (os.cpu_count() or 1) * 2: break
        t = time.perf_counter(); out = parallel(files, w); tp = time.perf_counter() - t
        assert out == ref
        print(f"Paralel {w} proses: {tp:.2f} s | speedup {t_seq/tp:.2f}x | efisiensi {t_seq/tp/w*100:.0f}%")
        res["par"][w] = tp
    json.dump(res, open("result.json", "w"))