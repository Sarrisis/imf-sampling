from Simulated_clusters_ran import generate_random_cluster
from Simulated_clusters_ran import summarize_cluster
from Simulated_clusters_opt import generate_optimal_cluster

import numpy as np
import pandas as pd


n_samples = 10000
target_masses = np.logspace(np.log10(50), np.log10(10000), n_samples)

ml_results = []

bins = np.logspace(np.log10(0.08), np.log10(150), 11)

for i, M in enumerate(target_masses):
    #  1. RANDOM CLUSTER
    m_rand = generate_random_cluster(M)
    s_rand = summarize_cluster(m_rand, bins=bins)
    s_rand["label"] = 0
    s_rand["target_mass"] = M
    ml_results.append(s_rand)

    # 2. OPTIMAL CLUSTER
    m_opt = generate_optimal_cluster(M)
    s_opt = summarize_cluster(m_opt, bins=bins)
    s_opt["label"] = 1
    s_opt["target_mass"] = M
    ml_results.append(s_opt)


df_ml = pd.DataFrame(ml_results)
df_ml.to_csv("ML_train_dataset_balanced.csv", index=False)