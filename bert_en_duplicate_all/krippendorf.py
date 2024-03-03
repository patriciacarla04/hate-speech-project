import numpy as np
import pandas as pd
from collections import defaultdict
import math

def encode_labels(label, values):
    return values.index(label)

def create_reliability_matrix(df, annotator_column1, annotator_column2, annotation_column1, annotation_column2):
    columns = sorted(list(set(df[annotator_column1])))
    #values = sorted(list(set(df[annotation_column1])))
    rl_matrix = defaultdict(list)
    for index, row in df.iterrows():
        rl_matrix[row[annotator_column1]].append(row[annotation_column1])
        rl_matrix[row[annotator_column2]].append(row[annotation_column2])
        for c in columns:
            if (c is not row[annotator_column1]) and (c is not row[annotator_column2]):
                rl_matrix[c].append(None)

    rl_matrix = pd.DataFrame(data=rl_matrix)
    return rl_matrix


def create_coincidence_matrix(rel_matrix, values):
    coincidence_matrix = np.zeros((len(values), len(values)))   # initialize
    total = 0
    for index, row in rel_matrix.iterrows():   # rows are the comments
        total += row.count()         # num of non-zeros in row
        for c in values:            # values are probably 0, 1, 2, 3
            for k in values:        # c and k are both values, indices of the coincidence matrix
                count = 0
                value_count = 0
                for column in rel_matrix.columns:  # column is annotator
                    if not pd.isna(row[column]):
                        value_count += 1
                    if row[column] == c:      # the comment of the annotator eqals to value
                        for column2 in rel_matrix.columns:
                            if column == column2:     # if the two annotations are the same, do nothing
                                continue
                            if row[column2] == k:     # count the disagreement c-k
                                count += 1
                #value_count = value_count * (value_count - 1)
                coincidence = count / (value_count - 1) # c-k disagreement / number of non-NAN values
                coincidence_matrix[c][k] += coincidence
    return coincidence_matrix, total


def alpha_score_from_coincidence(matrix, sum):
    row_sums = matrix.sum(axis=1)
    print(row_sums)
    #sum = row_sums.sum(axis=0)

    Do = sum - 1
    De = sum * (sum - 1)
    cc_coincidence = 0
    for c in range(matrix.shape[0]):
        print(c)
        cc_coincidence += matrix[c][c]
        print("CC_coincidence: {}, sum: {}".format(cc_coincidence, matrix[c][c]))

    Ae = 0
    for s in row_sums:
        Ae += s * (s - 1)

    Do = Do * cc_coincidence - Ae
    De = De - Ae
    print(Do)
    print(De)
    alpha = Do / De
    return alpha


def alpha_score(matrix):
    coincidence = matrix + np.transpose(matrix)
    row_sums = coincidence.sum(axis=1)
    print(row_sums)
    sum = row_sums.sum(axis=0)

    Do = 0
    De = 0
    for c in range(coincidence.shape[0]):
        print(c)
        for k in range(coincidence.shape[0] - 1, 0, -1):
            print(k)
            if k <= c:
                break
            Do += coincidence[c][k]
            De += row_sums[c] * row_sums[k]
    print(Do)
    print(De)
    alpha = 1 - (sum - 1) * (Do / De)
    return alpha


class KAlpha():
    def metric(self, matrix):
        pass

    def alpha_score(self, matrix):
        coincidence = matrix + np.transpose(matrix)
        row_sums = coincidence.sum(axis=1)
        #print(coincidence)
        sum = row_sums.sum(axis=0)
        metric_matrix = self.metric(coincidence)

        Do = 0
        De = 0
        for c in range(coincidence.shape[0]):
            #print(c)
            for k in range(coincidence.shape[0] - 1, 0, -1):
                #print(k)
                if k <= c:
                    break
                Do += coincidence[c][k] * metric_matrix[c][k]
                De += row_sums[c] * row_sums[k] * metric_matrix[c][k]
        #print(Do)
        #print(De)
        alpha = 1 - (sum - 1) * (Do / De)
        return alpha


class OrdinalKAlpha(KAlpha):
    def metric(self, matrix):
        metric_matrix = np.zeros_like(matrix)
        row_sums = matrix.sum(axis=1)
        for c in range(metric_matrix.shape[0]):
            for k in range(metric_matrix.shape[0]):
                #row = matrix[c]
                #print(row[c:k+1])
                if c <= k:
                    w = row_sums[c:k+1].sum()
                else:
                    w = row_sums[k:c+1].sum()
                w -= ((row_sums[c] + row_sums[k]) / 2)
                metric_matrix[c][k] = w
        #print(metric_matrix)
        return metric_matrix


class IntervalKAlpha(KAlpha):
    def metric(self, matrix):
        metric_matrix = np.zeros_like(matrix)
        for c in range(metric_matrix.shape[0]):
            for k in range(metric_matrix.shape[0]):
                metric_matrix[c][k] = math.pow((c-k), 2)
        #print(metric_matrix)
        return metric_matrix


class NominalKAlpha(KAlpha):
    def metric(self, matrix):
        metric_matrix = np.zeros_like(matrix)
        for c in range(metric_matrix.shape[0]):
            for k in range(metric_matrix.shape[0]):
                if c != k:
                    metric_matrix[c][k] = 1
        return metric_matrix
