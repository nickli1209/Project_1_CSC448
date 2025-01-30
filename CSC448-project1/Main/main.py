# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align.substitution_matrices import load
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from multiprocessing import Pool

# Constants
blosum62 = load("BLOSUM62")
gap_penalty = -12
scoring_matrix = blosum62
print(scoring_matrix)


# step 1, smith waterman algorithm

def smith_waterman(seq1: str, seq2: str):
    smith_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    max_score = 0
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            # not handling if no matching in blosum62, change to check both isdes of diag
            match_score = scoring_matrix.get((seq1[i - 1], seq2[j - 1]),
                                             scoring_matrix.get((seq2[j - 1], seq1[i - 1]), -1))
            smith_matrix[i][j] = max(0,
                                     smith_matrix[i - 1][j - 1] + match_score,
                                     int(smith_matrix[i - 1][j] + gap_penalty),
                                     int(smith_matrix[i][j - 1] + gap_penalty)
                                     )
            max_score = max(max_score, int(smith_matrix[i][j]))
    return max_score


def test_smith_waterman():
    seq1 = "ACAC"
    seq2 = "AC"
    score = smith_waterman(seq1, seq2)
    print(score)


test_smith_waterman()


def get_sequences(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]


def build_similarity_matrix(sequences):
    num_seqs = len(sequences)
    similarity_matrix = np.zeros((num_seqs, num_seqs))
    for i in range(num_seqs):
        for j in range(num_seqs):
            score = smith_waterman(sequences[i], sequences[j])
            similarity_matrix[i][j] = score
            similarity_matrix[j][i] = score
    return similarity_matrix


def compute_similarity_row(args):
    row_index, sequences = args
    num_seqs = len(sequences)
    row_scores = np.zeros(num_seqs)
    for j in range(num_seqs):
        row_scores[j] = smith_waterman(sequences[row_index], sequences[j])
    return row_index, row_scores


def build_similarity_matrix_parallel(sequences):
    num_seqs = len(sequences)
    similarity_matrix = np.zeros((num_seqs, num_seqs))
    args = [(i, sequences) for i in range(num_seqs)]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(compute_similarity_row, args)
    for row_index, row_scores in results:
        similarity_matrix[row_index] = row_scores
    return similarity_matrix


def plot_similarity_heatmap(sim_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis", square=True)
    plt.title("Similarity Matrix Heatmap")
    plt.xlabel("Sequences")
    plt.ylabel("Sequences")
    plt.savefig("Heatmap_2.png")
    plt.close()
    print("Heatmap saved as Heatmap_2.png")


def cluster_sequences(sim_matrix, seqs):
    linkage_matrix = linkage(sim_matrix, method='complete')
    # Plot the dendrogram
    labels = [f"seq{i + 1}" for i in range(len(seqs))]
    plt.figure(figsize=(10, 8))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Phylogenetic Tree (UPGMA)")
    plt.xlabel("Sequences")
    plt.ylabel("Distance")
    plt.savefig("phylo_tree_2.png")
    plt.close()
    print(f"Dendrogram saved as phylo_tree_2.png")


def normalize_similarity_matrix(similarity_matrix, sequences):
    num_seqs = len(sequences)
    normalized_matrix = np.zeros_like(similarity_matrix, dtype=float)

    for i in range(num_seqs):
        for j in range(num_seqs):
            if i != j:  # Avoid division by zero for diagonal
                avg_length = (len(sequences[i]) + len(sequences[j])) / 2
                normalized_matrix[i][j] = similarity_matrix[i][j] / avg_length
            else:
                normalized_matrix[i][j] = similarity_matrix[i][j]  # Keep diagonal as-is (or normalize if desired)

    return normalized_matrix


def find_closest_and_farthest_proteins_simple(normalized_matrix):
    num_seqs = len(normalized_matrix)
    pairs = [
        (i, j, normalized_matrix[i][j])
        for i in range(num_seqs)
        for j in range(i + 1, num_seqs)  # Only upper triangle, no diagonal
    ]
    # Sort pairs by score
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    two_farthest = pairs_sorted[:2]  # Smallest scores
    two_closest = pairs_sorted[-2:]  # Largest scores (reverse order)

    return two_closest[::-1], two_farthest


if __name__ == "__main__":
    pass
    # # builds similarity matrix, takes forever
    # sequences = get_sequences("seq.txt")
    # print(sequences[0])
    # # single processor, old method
    # # similarity_matrix = build_similarity_matrix(sequences)
    # # parrallel solution for building similarity matrix
    # similarity_matrix = build_similarity_matrix_parallel(sequences)
    # # normalize the similarity matrix
    # normalized_matrix = normalize_similarity_matrix(similarity_matrix, sequences)
    # # plot heatmap and dendrogram
    # plot_similarity_heatmap(similarity_matrix)
    # cluster_sequences(similarity_matrix, sequences)
    # # find closest and furthest proteins
    # closest_two = find_closest_and_farthest_proteins_simple(normalized_matrix)[0]
    # farthest_two = find_closest_and_farthest_proteins_simple(normalized_matrix)[1]
    # print(f"Closest two proteins: {closest_two}")
    # print(f"Farthest two proteins: {farthest_two}")
