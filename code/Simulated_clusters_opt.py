import csv
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "gal",
    "/Users/karel/PycharmProjects/galIMF/galIMF_version_1.0.py"
)
gal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gal)


def mass_to_luminosity(m):
    if m < 0.43:
        return 0.23 * m**2.3
    elif m < 2.0:
        return m**4
    elif m < 55.0:
        return 1.4 * m**3.5
    else:
        return 32000 * m


def generate_optimal_cluster(M_ecl):
    gal.function_sample_from_IMF(
        M_ecl,
        1,
        0.08,
        1.3,
        0.5,
        2.3,
        1.0,
        2.3,
        150
    )

    masses = []

    for m, n in zip(gal.list_M_str_i, gal.list_n_str_i):
        masses.extend([float(m)] * int(n))

    return masses

def summarize_cluster(masses, bins=None, thresholds=None):
    if bins is None:
        bins = np.logspace(np.log10(0.08), np.log10(150), 11)

    if thresholds is None:
        thresholds = [0.5, 1, 2, 4, 8, 20]

    if len(masses) == 0:
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
            name = str(t).replace(".", "p")
            result[f"N_above_{name}"] = 0
            result[f"mass_fraction_above_{name}"] = 0.0

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

    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (masses >= bins[i]) & (masses <= bins[i + 1])
        else:
            mask = (masses >= bins[i]) & (masses < bins[i + 1])

        bin_count = np.sum(mask)
        bin_mass = masses[mask].sum()

        result[f"bin_{i + 1}_count"] = int(bin_count)

        # početní frakce
        result[f"bin_{i + 1}_fraction"] = (
            float(bin_count / n_stars) if n_stars > 0 else 0.0
        )

        # hmotnostní frakce
        result[f"bin_{i + 1}_mass_fraction"] = (
            float(bin_mass / total_mass) if total_mass > 0 else 0.0
        )
    return result

if __name__ == "__main__":
    #total_realizations = 600
    #target_masses = np.logspace(np.log10(50), np.log10(10000), total_realizations)
    target_masses = [50, 100, 300, 1000, 3000, 10000]
    total_realizations = len(target_masses)

    bins = np.logspace(np.log10(0.08), np.log10(150), 11)
    thresholds = [0.5, 1, 2, 4, 8, 20]

    results = []

    for i, M in enumerate(target_masses):
        masses = generate_optimal_cluster(M)
        summary = summarize_cluster(masses, bins=bins, thresholds=thresholds)

        summary["target_mass"] = float(M)
        summary["cluster_id"] = i
        results.append(summary)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{total_realizations} realizations (M_ecl = {M:.1f})")

    if results:
        fieldnames = list(results[0].keys())

        with open("../../optimal_sampling_clusters6.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("Data are saved as optimal_sampling_clusters6.csv")
