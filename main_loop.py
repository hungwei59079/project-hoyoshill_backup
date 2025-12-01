import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os
import csv
import sys


N_list = [8, 16, 32, 64, 128, 256, 512, 1024]
beta_list = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 48, 64, 90, 128, 170, 256, 384, 512]
n_steps = 500 # number of Monte Carlo steps
thermalization_steps = 100 # number of thermalization steps
sample_steps = 5 # Measurment interval

N = N_list[int(sys.argv[1])]
L = int(N/2) # initial length of operator string
n_b = N - 1

os.makedirs("./output/csv/", exist_ok=True)
os.makedirs("./output/npz/", exist_ok=True)

results = {
    "beta": [],
    "n": [],
    "n2":[],
    "E": [],
    "Cv": [],
    "Cr": [],
    "Ms": [],
    "chi_s": [],
}

for beta in beta_list:
    print(f"N={N}, beta={beta}")
    L = int(N / 2) # initial length of operator string
    n_b = N - 1

    spin_config = np.array([1] * (N // 2) + [-1] * (N // 2))
    np.random.shuffle(spin_config)
    op_index_sequence = [0] * L

    for i in range(thermalization_steps):
        diagonal_update(op_index_sequence, spin_config, n_b, beta)
        v_list, Vfirst = vertex_list(op_index_sequence, N)
        loop_update(op_index_sequence, v_list, spin_config, Vfirst)
        n = np.count_nonzero(op_index_sequence)
        if L < n * 2:
            L = int(n * 2)
            op_index_sequence.extend([0] * (L - len(op_index_sequence)))

    n_list = []
    n2_list = []
    Cr_list = []
    Ms_list = []
    chi_s_list = []

    for i in range(n_steps):
        diagonal_update(op_index_sequence, spin_config, n_b, beta)
        v_list, Vfirst = vertex_list(op_index_sequence, N)
        loop_update(op_index_sequence, v_list, spin_config, Vfirst)
        if i % sample_steps == 0:
            n, Cr, Ms, chi_s = measurement(op_index_sequence, spin_config, beta)
            n_list.append(n)
            n2_list.append(n**2)
            Cr_list.append(Cr)
            Ms_list.append(Ms)
            chi_s_list.append(chi_s)
    n = np.mean(n_list)
    n2 = np.mean(n2_list)
    E = -np.mean(n_list) / beta / N
    Cv = (np.mean(np.array(n_list)**2) - np.mean(n_list)**2 -np.mean(n_list))/N
    Cr = np.mean(Cr_list, axis=0)
    Ms = np.mean(Ms_list)
    chi_s = np.mean(chi_s_list)

    results["beta"].append(beta)
    results["n"].append(n)
    results["n2"].append(n2)
    results["E"].append(E)
    results["Cv"].append(Cv)
    results["Cr"].append(Cr)
    results["Ms"].append(Ms)
    results["chi_s"].append(chi_s)

csv_path = f"./output/csv/1D_Heisenberg_{N}.csv"
with open(csv_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["beta", "n", "n2", "E", "Cv", "Ms", "chi_s", "Cr"])
    for i in range(len(results["beta"])):
        writer.writerow([
            results["beta"][i],
            results["n"][i],
            results["n2"][i],
            results["E"][i],
            results["Cv"][i],
            results["Ms"][i],
            results["chi_s"][i],
            results["Cr"][i].tolist()  # Cr is an array
        ])

print(f"Saved CSV to {csv_path}")

    # ================================
    # Save NPZ
    # ================================
npz_path = f"./output/npz/1D_Heisenberg_{N}.npz"
np.savez(
    npz_path,
    beta=np.array(results["beta"]),
    n=np.array(results["n"]),
    n2=np.array(results["n2"]),
    E=np.array(results["E"]),
    Cv=np.array(results["Cv"]),
    Cr=np.array(results["Cr"]),
    Ms=np.array(results["Ms"]),
    chi_s=np.array(results["chi_s"]),
)

print(f"Saved NPZ to {npz_path}")