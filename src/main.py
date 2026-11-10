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

#=============================================================================
#БЛОК 2: Варіант 2 (MTBF сервера, Exp(0.05))
#=============================================================================

def run_block_2():
    print("=== БЛОК 2: Варіант 2 (MTBF сервера, Exp(0.05)) ===")
    N = 100000
    lmbda = 0.05
    
    # Генеруємо вибірку для показникового розподілу
    sample = np.random.exponential(1 / lmbda, N)
    
    # Теоретичні значення
    theor_M = 1 / lmbda
    theor_D = 1 / (lmbda**2)
    theor_sigma = 1 / lmbda
    theor_P = np.exp(-lmbda * 30) # P(X > 30)
    
    # Практичні (вибіркові) значення
    emp_M = np.mean(sample)
    emp_D = np.var(sample)
    emp_sigma = np.std(sample)
    emp_P = np.mean(sample > 30) # Частка елементів, більших за 30
    
    print(f"M[X]:     Практика = {emp_M:.4f} | Теорія = {theor_M:.4f}")
    print(f"D[X]:     Практика = {emp_D:.4f} | Теорія = {theor_D:.4f}")
    print(f"sigma[X]: Практика = {emp_sigma:.4f} | Теорія = {theor_sigma:.4f}")
    print(f"P(X > 30):Практика = {emp_P:.4f} | Теорія = {theor_P:.4f}\n")

    # Графік щільності f(x)
    plt.figure()
    plt.hist(sample, bins=80, density=True, alpha=0.6, color='green', label='Гістограма')
    x = np.linspace(0, max(sample), 200)
    f_x = lmbda * np.exp(-lmbda * x)
    plt.plot(x, f_x, 'r--', label='Теорія f(x)')
    plt.title("Блок 2: Щільність ймовірності (MTBF)")
    plt.xlabel("Час, год")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphics/block2_exp_mtbf.png", dpi=300)
    plt.close()

#=============================================================================
#БЛОК 3: Перцентильний аналіз затримок (Latency Tail)
#=============================================================================

def run_block_3():
    print("=== БЛОК 3: Перцентильний аналіз затримок (Latency Tail) ===")
    N = 100000
    
    # Створюємо масив затримок: 90% звичайні (Normal), 10% важкі (Exp)
    latencies = []
    for _ in range(N):
        if random.random() < 0.90:
            val = random.gauss(20, 4) # Нормальний N(20, 4^2)
        else:
            val = random.expovariate(0.01) # Показниковий Exp(0.01)
        latencies.append(val)
        
    latencies = np.array(latencies)
    latencies = np.maximum(latencies, 0) # Прибираємо від'ємні значення
    
    # Обчислюємо перцентилі
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Середнє M[X] = {np.mean(latencies):.2f} мс")
    print(f"p50 (медіана) = {p50:.2f} мс")
    print(f"p90           = {p90:.2f} мс")
    print(f"p95           = {p95:.2f} мс")
    print(f"p99           = {p99:.2f} мс\n")

    # Графік затримок
    plt.figure()
    plt.hist(latencies, bins=100, density=True, alpha=0.6, color='purple')
    plt.axvline(p50, color='g', linestyle='--', label=f'p50 = {p50:.1f}')
    plt.axvline(p90, color='b', linestyle='--', label=f'p90 = {p90:.1f}')
    plt.axvline(p95, color='orange', linestyle='--', label=f'p95 = {p95:.1f}')
    plt.axvline(p99, color='r', linestyle='--', label=f'p99 = {p99:.1f}')
    plt.xlim(0, 250)
    plt.title("Блок 3: Розподіл затримок ПЗ")
    plt.xlabel("Затримка, мс")
    plt.ylabel("Щільність")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphics/block3_latency_tail.png", dpi=300)
    plt.close()

#=============================================================================
#БЛОК 4: Варіант 8 (Тривалість сесії, Exp(0.01))
#=============================================================================

def run_block_4():
    print("=== БЛОК 4: Варіант 8 (Тривалість сесії, Exp(0.01)) ===")
    N = 100000
    lmbda = 0.01
    
    # Генеруємо вибірку сесій
    sessions = np.random.exponential(1 / lmbda, N)
    
    # Теоретичні значення для p50 та p90
    theor_p50 = np.log(2) / lmbda        # ~69.31 с
    theor_p90 = -np.log(0.10) / lmbda    # ~230.26 с
    
    # Практичні значення
    emp_p50 = np.percentile(sessions, 50)
    emp_p90 = np.percentile(sessions, 90)
    
    print(f"p50: Практика = {emp_p50:.2f} с | Теорія = {theor_p50:.2f} с")
    print(f"p90: Практика = {emp_p90:.2f} с | Теорія = {theor_p90:.2f} с\n")

    plt.figure()
    plt.hist(sessions, bins=100, density=True, alpha=0.6, color='teal')
    plt.axvline(emp_p50, color='g', linestyle='--', label=f'p50 = {emp_p50:.1f} с')
    plt.axvline(emp_p90, color='r', linestyle='--', label=f'p90 = {emp_p90:.1f} с')
    plt.xlim(0, 500)
    plt.title("Блок 4: Тривалість сесії користувача")
    plt.xlabel("Час, с")
    plt.ylabel("Щільність")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphics/block4_user_session.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_block_1()
    run_block_2()
    run_block_3()
    run_block_4()