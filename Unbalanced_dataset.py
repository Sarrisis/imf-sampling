from My_project.Thesis.Simulated_clusters_ran import generate_random_cluster
from My_project.Thesis.Simulated_clusters_ran import summarize_cluster
from My_project.Thesis.code.Simulated_clusters_opt import generate_optimal_cluster

import numpy as np
import pandas as pd

n_masses = 1000
n_random_per_mass = 20
target_masses = np.logspace(np.log10(50), np.log10(10000), n_masses)
bins = np.logspace(np.log10(0.08), np.log10(150), 11)
ml_results = []

for i, M in enumerate(target_masses):

    # OPTIMAL CLUSTER
    m_opt = generate_optimal_cluster(M)
    s_opt = summarize_cluster(m_opt, bins=bins)
    s_opt["label"] = 1          # 1 = optimal
    s_opt["target_mass"] = M
    s_opt["realization_id"] = 0
    s_opt["sampling"] = "optimal"
    ml_results.append(s_opt)

    # RANDOM CLUSTERS
    for j in range(n_random_per_mass):
        m_rand = generate_random_cluster(M)
        s_rand = summarize_cluster(m_rand, bins=bins)
        s_rand["label"] = 0      # 0 = random
        s_rand["target_mass"] = M
        s_rand["realization_id"] = j
        s_rand["sampling"] = "random"
        ml_results.append(s_rand)


df_ml = pd.DataFrame(ml_results)

df_ml.to_csv("ML_train_dataset_unbalanced.csv", index=False)

print(df_ml.shape)
print(df_ml["sampling"].value_counts())