import gzip
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_fastq(filename):
    with gzip.open(filename, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline().strip()
            qual = f.readline().strip()
            yield seq, qual


def quality_to_phred(quality_string):
    return [ord(char) - 33 for char in quality_string]


def analyze_fastq(filename):
    read_counts = 0
    lengths = []
    all_mean_quals = []
    q20_count = 0
    q30_count = 0
    per_base_accumulator = []

    for seq, qual in parse_fastq(filename):
        read_counts += 1
        lengths.append(len(seq))

        phred_scores = quality_to_phred(qual)
        mean_q = np.mean(phred_scores)

        if mean_q > 20: q20_count += 1
        if mean_q > 30: q30_count += 1

        all_mean_quals.append(mean_q)
        per_base_accumulator.append(phred_scores)

    unique_lengths = set(lengths)
    if len(unique_lengths) == 1:
        len_output = f"{lengths[0]} bp (constant)"
    else:
        len_output = f"min:{min(lengths)}, max:{max(lengths)}, mean:{np.mean(lengths):.1f}"

    print("=== FASTQ FILE ANALYSIS ===")
    print(f"File: {filename}")
    print(f"Number of reads: {read_counts}")
    print(f"Read length: {len_output}")
    print("\nQuality statistics:")
    print(f"  Mean quality: {np.mean(all_mean_quals):.1f}")
    print(f"  Reads with Q > 30: {q30_count} ({q30_count / read_counts * 100:.1f}%)")
    print(f"  Reads with Q > 20: {q20_count} ({q20_count / read_counts * 100:.1f}%)")

    return per_base_accumulator



def plot_per_base_quality(per_base_data):
    plt.figure(figsize=(12, 6))

    plt.axhspan(28, 40, color='green', alpha=0.2)
    plt.axhspan(20, 28, color='orange', alpha=0.2)
    plt.axhspan(0, 20, color='red', alpha=0.2)

    plt.boxplot(per_base_data, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='yellow', color='black'),
                medianprops=dict(color='red'))

    plt.title("Per Base Sequence Quality")
    plt.xlabel("Position in read (bp)")
    plt.ylabel("Quality Score (Phred)")
    plt.ylim(0, 40)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('per_base_quality.png')
    print("\nSaved plot: per_base_quality.png")


def main():
    fastq_file = 'sample_data.fastq.gz'

    if not os.path.exists(fastq_file):
        print(f"Error: {fastq_file} not found.")
        return

    quality_data = analyze_fastq(fastq_file)
    plot_per_base_quality(quality_data)


if __name__ == "__main__":
    main()

#Answering to: How does quality change along the read?
# Interpretation: Quality at the first time is high, but once we proceed, it tends to drop due to some chemistry aspects.