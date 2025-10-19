import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data = {
    1: [2.0, 2.8, 3.3, 3.2, 4.4, 3.6, 2.9, 2.5, 2.8, 2.1],
    2: [3.3, 3.6, 2.6, 3.1, 3.2, 3.3, 2.9, 3.4, 3.2, 3.2],
    3: [3.2, 3.3, 3.2, 2.9, 3.3, 2.5, 2.6, 2.8],
    4: [3.5, 2.8, 3.2, 3.5, 2.3, 2.4, 2.0, 1.6],
    5: [2.6, 2.6, 2.9, 2.0, 2.0, 2.1],
    6: [3.1, 2.9, 3.1, 2.5],
    7: [2.6, 2.2, 2.2, 2.5, 1.2, 1.2],
    8: [2.5, 2.4, 3.0, 1.5]
}

group_means = [np.mean(data[i]) for i in range(1, len(data) + 1)]
n_j = [len(data[i]) for i in range(1, 9)]

all_values = []
for i in range(1, len(data) + 1):
    all_values.extend(data[i])
mean_total = np.mean(all_values)

SSH = 0
for i in range(len(data)):
    SSH += n_j[i] * (group_means[i] - mean_total) ** 2  # noqa

SSE = 0
for i in range(len(data)):
    group_data = data[i + 1]
    group_mean = group_means[i]
    SSE += sum((x - group_mean) ** 2 for x in group_data)  # noqa

n = len(all_values)
t = 7
r = 8

SSH_normalized = SSH / t
SSE_normalized = SSE / (n - r)

F_statistic = SSH_normalized / SSE_normalized
p_value = 1 - stats.f.cdf(F_statistic, t, n - r)
alpha = 0.05

print("(I) Проверка гипотезы о совпадении математических ожиданий:")
print(f"    Нормированное значение SSH: {SSH_normalized:.4f}")
print(f"    Нормированное значение SSE: {SSE_normalized:.4f}")
print(f"    F-статистика: {F_statistic:.4f}")
print(f"    p-value: {p_value:.4f}")
print(f"    Гипотеза о совпадении математических ожиданий должна быть отвергнута" if p_value < alpha else "Нет оснований отвергать гипотезу о совпадении математических ожиданий")

print("\n(II) S-метод:")

gamma_star = (1 / 3) * (group_means[0] + group_means[1] + group_means[2]) - (1 / 5) * (group_means[3] + group_means[4] + group_means[5] + group_means[6] + group_means[7])  # noqa
sum_3_5_cj2_nj = (1 / 9) * (1 / n_j[0] + 1 / n_j[1] + 1 / n_j[2]) + (1 / 25) * (1 / n_j[3] + 1 / n_j[4] + 1 / n_j[5] + 1 / n_j[6] + 1 / n_j[7])
D_gamma_star_3_5 = SSE_normalized * sum_3_5_cj2_nj

F_critical_95 = stats.f.ppf(0.95, t, n - r)
margin_95 = np.sqrt(t * F_critical_95 * D_gamma_star_3_5)

print(f"     95% ДИ для сравнения опоросов 1-3 с 4-8: ({gamma_star - margin_95:.4f}, {gamma_star + margin_95:.4f})")

gamma2_star = (1 / 4) * (group_means[0] + group_means[1] + group_means[2] + group_means[3]) - (1 / 4) * (group_means[4] + group_means[5] + group_means[6] + group_means[7])  # noqa
sum_4_4_cj2_nj = (1 / 16) * (1 / n_j[0] + 1 / n_j[1] + 1 / n_j[2] + 1 / n_j[3]) + (1 / 16) * (1 / n_j[4] + 1 / n_j[5] + 1 / n_j[6] + 1 / n_j[7])
D_gamma_star_4_4 = SSE_normalized * sum_4_4_cj2_nj

F_critical_90 = stats.f.ppf(0.9, t, n - r)
margin_90 = np.sqrt(t * F_critical_90 * D_gamma_star_4_4)

print(f"     90% ДИ для сравнения опоросов 1-4 с 5-8: ({gamma2_star - margin_90:.4f}, {gamma2_star + margin_90:.4f})")

plt.figure(figsize=(10, 6))
for group in range(1, 9):
    group_data = data[group]
    x_pos = [group] * len(group_data)
    plt.scatter(x_pos, group_data, alpha=0.75)

plt.scatter(range(1, 9), group_means, color='red', s=40, marker='D', label='Средние по группам')
plt.axhline(y=mean_total, color='black', linestyle='--', label='Общее среднее')  # noqa
plt.xlabel('Группы')
plt.ylabel('Вес поросят')
plt.title('График разброса веса поросят по группам')
plt.xticks(range(1, 9))
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()