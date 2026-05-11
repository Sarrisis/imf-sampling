import csv
import os
import numpy as np


def sample_kroupa_piecewise(n=1):
    m_min, m_mid, m_max = 0.08, 0.5, 150.0
    a1, a2 = 1.3, 2.3
    c1, c2 = 2.0, 1.0

    i1 = c1 * (m_mid ** (1 - a1) - m_min ** (1 - a1)) / (1 - a1)
    i2 = c2 * (m_max ** (1 - a2) - m_mid ** (1 - a2)) / (1 - a2)
    p_segment1 = i1 / (i1 + i2)

    u = np.random.random(n)
    masses = np.zeros(n)

    mask1 = u < p_segment1
    if np.any(mask1):
        u1 = u[mask1] / p_segment1
        masses[mask1] = (u1 * (m_mid ** (1 - a1) - m_min ** (1 - a1)) + m_min ** (1 - a1)) ** (1 / (1 - a1))

    mask2 = ~mask1
    if np.any(mask2):
        u2 = (u[mask2] - p_segment1) / (1 - p_segment1)
        masses[mask2] = (u2 * (m_max ** (1 - a2) - m_mid ** (1 - a2)) + m_mid ** (1 - a2)) ** (1 / (1 - a2))

    return masses

import numpy as np

def mass_to_luminosity(m):
    m = np.atleast_1d(m)
    l = np.zeros_like(m, dtype=float)

    mask1 = m < 0.43
    l[mask1] = 0.23 * m[mask1]**2.3

    mask2 = (m >= 0.43) & (m < 2.0)
    l[mask2] = m[mask2]**4

    mask3 = (m >= 2.0) & (m < 55.0)
    l[mask3] = 1.4 * m[mask3]**3.5

    mask4 = m >= 55.0
    l[mask4] = 32000 * m[mask4]

    return l

def generate_random_cluster(target_mass, tolerance=0.1):
    masses = []
    current_mass = 0.0

    while (target_mass - current_mass) > tolerance:
        m = sample_kroupa_piecewise(1)[0]

        if current_mass + m <= target_mass:
            masses.append(m)
            current_mass += m

    return masses

def summarize_cluster(masses, bins=None, thresholds=None):
    if bins is None:
        bins = np.logspace(np.log10(0.08), np.log10(150), 11)

    if thresholds is None:
        thresholds = [0.5, 1, 2, 4, 8, 20]

    if not masses:
        result = {
            "realized_mass": 0.0,
            "n_stars": 0,
            "m_max": 0.0,
            "mean_mass": 0.0,
            "n_above_8": 0,
            "n_above_20": 0,
            "L_total": 0.0,
            "max_luminosity": 0.0,
            "luminosity_ratio": 0.0,
            "m1": 0.0,
            "m2": 0.0,
            "m3": 0.0,
            "m1_minus_m2": 0.0,
            "m2_minus_m3": 0.0,
            "m1_over_m2": 0.0,
            "top3_mass_fraction": 0.0,
            "top5_mass_fraction": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q90_minus_q10": 0.0,
            "q75_over_q25": 0.0,
            "std_mass": 0.0,
            "var_mass": 0.0,
            "iqr_mass": 0.0,
        }

        for t in thresholds:
            result[f"N_above_{str(t).replace('.', 'p')}"] = 0
            result[f"mass_fraction_above_{str(t).replace('.', 'p')}"] = 0.0

        for i in range(len(bins) - 1):
            result[f"bin_{i+1}_count"] = 0
            result[f"bin_{i+1}_fraction"] = 0.0

        return result

    masses = np.array(masses, dtype=float)
    sorted_masses_desc = np.sort(masses)[::-1]
    luminosities = np.array([mass_to_luminosity(m) for m in masses], dtype=float)

    total_mass = masses.sum()
    n_stars = len(masses)
    total_luminosity = luminosities.sum()
    max_luminosity = luminosities.max()

    m1 = sorted_masses_desc[0] if n_stars >= 1 else 0.0
    m2 = sorted_masses_desc[1] if n_stars >= 2 else 0.0
    m3 = sorted_masses_desc[2] if n_stars >= 3 else 0.0

    q10, q25, q50, q75, q90 = np.percentile(masses, [10, 25, 50, 75, 90])

    counts, _ = np.histogram(masses, bins=bins)

    result = {
        "realized_mass": float(total_mass),
        "n_stars": int(n_stars),
        "m_max": float(masses.max()),
        "mean_mass": float(masses.mean()),
        "n_above_8": int(np.sum(masses > 8)),
        "n_above_20": int(np.sum(masses > 20)),
        "L_total": float(total_luminosity),
        "max_luminosity": float(max_luminosity),
        "luminosity_ratio": float(max_luminosity / total_luminosity) if total_luminosity > 0 else 0.0,
        "m1": float(m1),
        "m2": float(m2),
        "m3": float(m3),
        "m1_minus_m2": float(m1 - m2),
        "m2_minus_m3": float(m2 - m3),
        "m1_over_m2": float(m1 / m2) if m2 > 0 else 0.0,
        "top3_mass_fraction": float(sorted_masses_desc[:3].sum() / total_mass) if total_mass > 0 else 0.0,
        "top5_mass_fraction": float(sorted_masses_desc[:5].sum() / total_mass) if total_mass > 0 else 0.0,
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "q90_minus_q10": float(q90 - q10),
        "q75_over_q25": float(q75 / q25) if q25 > 0 else 0.0,
        "std_mass": float(np.std(masses)),
        "var_mass": float(np.var(masses)),
        "iqr_mass": float(q75 - q25),
    }

    for t in thresholds:
        name = str(t).replace(".", "p")
        mask = masses > t
        result[f"N_above_{name}"] = int(np.sum(mask))
        result[f"mass_fraction_above_{name}"] = float(masses[mask].sum() / total_mass) if total_mass > 0 else 0.0

    for i, c in enumerate(counts):
        result[f"bin_{i+1}_count"] = int(c)
        result[f"bin_{i+1}_fraction"] = float(c / n_stars) if n_stars > 0 else 0.0

    return result


def simulate_cluster_grid(cluster_masses, n_clusters_each, bins=None, thresholds=None):
    results = []

    for target_mass in cluster_masses:
        for i in range(n_clusters_each):
            masses = generate_random_cluster(target_mass)
            summary = summarize_cluster(masses, bins=bins, thresholds=thresholds)

            summary["target_mass"] = target_mass
            summary["cluster_id"] = i
            results.append(summary)

            if i % 1000 == 0:
                print(f"M_ecl = {target_mass}, cluster {i}")

    return results

def save_results_to_csv(results, filename):
    if not results:
        return

    keys = results[0].keys()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    np.random.seed(42)

    cluster_masses = [50, 100, 300, 1000, 3000, 10000]
    n_clusters_each = 10000

    bins = np.logspace(np.log10(0.08), np.log10(150), 11)
    thresholds = [0.5, 1, 2, 4, 8, 20]

    results = simulate_cluster_grid(
        cluster_masses,
        n_clusters_each,
        bins=bins,
        thresholds=thresholds
    )

    print("Saving CSV to:", os.path.abspath("random_sampling_results.csv"))
    save_results_to_csv(results, "random_sampling_results.csv")
    print("Simulation finished.")
