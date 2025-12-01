import numpy as np
import os
import matplotlib.pyplot as plt

def load_results(N):
    npz_path = os.path.join("output", "npz", f"1D_Heisenberg_{N}.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    return data


if __name__ == "__main__":
    # Output directory for plots
    os.makedirs("output/plots", exist_ok=True)

    N_list = [8, 16]

    for N in N_list:
        data = load_results(N)

        beta = data["beta"]
        n = data["n"]
        n2 = data["n2"]
        E = data["E"]
        Cv = data["Cv"]
        Ms = data["Ms"]
        chi_s = data["chi_s"]

        print(f"Plotting results for N = {N}")

        # Create a figure with 5 subplots
        fig, axs = plt.subplots(7, 1, figsize=(8, 18), sharex=True)
        fig.suptitle(f"1D Heisenberg Model Results (N={N})", fontsize=16)

        # --- Subplot 1: β vs n ---
        axs[0].plot(beta, n, marker='o')
        axs[0].set_ylabel("n")
        axs[0].grid(True)

        axs[1].plot(beta, n2, marker='o')
        axs[1].set_ylabel("n2")
        axs[1].grid(True)

        # --- Subplot 2: β vs E ---
        axs[2].plot(beta, E, marker='o')
        axs[2].set_ylabel("E")
        axs[2].grid(True)

        # --- Subplot 3: β vs Cv ---
        axs[3].plot(beta, Cv, marker='o')
        axs[3].set_ylabel("Cv")
        axs[3].grid(True)

        # --- Subplot 4: β vs Ms ---
        axs[4].plot(beta, Ms, marker='o')
        axs[4].set_ylabel("Ms")
        axs[4].grid(True)

        # --- Subplot 5: β vs χ_s ---
        axs[5].plot(beta, chi_s, marker='o')
        axs[5].set_ylabel("chi_s")
        axs[5].set_xlabel("beta")
        axs[5].grid(True)

        # Save figure
        plot_path = f"output/plots/plot_1D_heisenberg_{N}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for title
        fig.savefig(plot_path)
        plt.close(fig)

        plt.plot(1/beta, E, marker='o')
        plt.xlabel("T")
        plt.ylabel("E")
        plt.savefig(f"output/plots/plot_1D_heisenberg_{N}_E_to_T.png")

        print(f"Saved plot to {plot_path}\n")