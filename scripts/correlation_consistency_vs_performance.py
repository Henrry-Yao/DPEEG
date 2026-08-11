#!/usr/bin/env python3
"""Trial consistency vs task performance -- Pearson/Spearman correlation (n=10)"""
import numpy as np
from scipy import stats

subjects = ["sub-%02d" % i for i in range(1, 11)]

# Inter-trial consistency (from compute_inter_trial_consistency.py)
consistency = [0.0828, 0.0530, 0.0964, 0.0880, 0.0369, 0.0632, 0.0995, 0.0900, 0.0392, 0.0893]

# Top-1 accuracy (from paper Table 7)
decoding       = [42.1, 51.2, 51.6, 51.8, 44.5, 57.7, 49.0, 57.5, 46.9, 63.0]
retrieval_enc  = [53.5, 53.0, 57.0, 54.5, 42.0, 60.0, 55.0, 65.0, 47.0, 60.0]
generative_enc = [ 9.5, 16.5, 15.5,  9.0, 35.0, 22.5, 12.5, 28.0, 70.5, 19.0]

def sig_str(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."

print("=" * 70)
print("Inter-trial Consistency vs Task Performance (n=10)")
print("=" * 70)

for name, vals in [("Decoding", decoding),
                   ("Retrieval Encoding", retrieval_enc),
                   ("Generative Encoding", generative_enc)]:
    r, p = stats.pearsonr(consistency, vals)
    print("  %-25s  r = %+.4f  p = %.4f  %s" % (name, r, p, sig_str(p)))

print()
print("%-10s %12s %10s %10s %12s" % ("Subject", "Consistency", "Gen Enc", "Decoding", "Retr Enc"))
print("-" * 60)
for i in range(10):
    print("%-10s %12.4f %10.1f %10.1f %12.1f" % (
        subjects[i], consistency[i], generative_enc[i], decoding[i], retrieval_enc[i]))

# Spearman
print()
print("Spearman rank correlation (more robust to outlier sub-09):")
for name, vals in [("Decoding", decoding),
                   ("Retrieval Encoding", retrieval_enc),
                   ("Generative Encoding", generative_enc)]:
    rho, p = stats.spearmanr(consistency, vals)
    print("  %-25s  rho = %+.4f  p = %.4f  %s" % (name, rho, p, sig_str(p)))

# Also: consistency vs itself check & partial correlation note
print()
print("Additional diagnostics:")
print("  Mean consistency:           %.4f" % np.mean(consistency))
print("  Mean gen encoding top-1:    %.1f" % np.mean(generative_enc))
print("  Consistency Std:            %.4f" % np.std(consistency, ddof=1))
print("  Gen encoding Std:           %.1f" % np.std(generative_enc, ddof=1))
