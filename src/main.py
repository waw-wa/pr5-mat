import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Фіксація генератора псевдовипадкових чисел для відтворюваності
random.seed(993)
np.random.seed(993)

# Створення папки для графіків
os.makedirs("graphics", exist_ok=True)

#=============================================================================
#БЛОК 1: Програмна перевірка Центральної Граничної Теореми
#=============================================================================

def run_block_1():
    print("=== БЛОК 1: Програмна перевірка Центральної Граничної Теореми ===")
    N = 100000
    k_values = [1, 2, 5, 12, 30]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, k in enumerate(k_values):
        # Сума k незалежних рівномірних НВВ U(0,1)
        samples = np.random.uniform(0, 1, (N, k))
        S_k = np.sum(samples, axis=1)
        
        # Стандартизація Z_k = (S_k - M[S_k]) / sqrt(D[S_k])
        Z_k = (S_k - 0.5 * k) / np.sqrt(k / 12.0)
        
        mean_Z = np.mean(Z_k)
        var_Z = np.var(Z_k)
        
        # Коефіцієнти асиметрії (Skewness) та ексцесу (Kurtosis)
        skew_Z = np.mean((Z_k - mean_Z)**3) / (var_Z**1.5)
        kurt_Z = (np.mean((Z_k - mean_Z)**4) / (var_Z**2)) - 3.0
        
        print(f"k={k:<2} | M[Z_k]={mean_Z:8.5f} | D[Z_k]={var_Z:7.5f} | Skewness={skew_Z:8.5f} | Kurtosis={kurt_Z:8.5f}")
        
        # Гістограма щільності
        axes[i].hist(Z_k, bins=80, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Теоретична щільність N(0,1)
        z_grid = np.linspace(-4, 4, 200)
        phi_z = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-z_grid**2 / 2.0)
        axes[i].plot(z_grid, phi_z, 'r--', linewidth=2, label=r'$\mathcal{N}(0, 1)$')
        
        axes[i].set_title(f"k = {k}")
        axes[i].set_xlabel("z")
        axes[i].set_ylabel("f(z)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig("graphics/block1_clt_simulation.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block1_clt_simulation.png\n")



if __name__ == "__main__":
    run_block_1()