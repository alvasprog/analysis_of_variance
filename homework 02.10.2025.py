import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import probplot

np.random.seed(hash("q Salimov Rustem Faridovich") % 10)


def mnk_estimation(x, y):
    size = len(x)
    X_matrix = np.column_stack([np.ones(size), x])
    beta_mnk_hat = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y
    return beta_mnk_hat[0], beta_mnk_hat[1]


def alt_estimation(x, y):
    size = len(x)
    third = size // 3

    X_lower = x[:third]
    Y_lower = y[:third]
    X_upper = x[size - third:]
    Y_upper = y[size - third:]

    X_lower_mean = X_lower.mean()
    Y_lower_mean = Y_lower.mean()
    X_upper_mean = X_upper.mean()
    Y_upper_mean = Y_upper.mean()

    beta1_alt_hat = (Y_upper_mean - Y_lower_mean) / (X_upper_mean - X_lower_mean)
    beta0_alt_hat = y.mean() - beta1_alt_hat * x.mean()

    return beta0_alt_hat, beta1_alt_hat


def mm_estimation(x, y):
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)

    beta1_mm_hat = cov_xy / var_x
    beta0_mm_hat = y.mean() - beta1_mm_hat * x.mean()

    return beta0_mm_hat, beta1_mm_hat


def coverage_probability(beta_hat, beta_true, alpha=0.05):
    se = np.std(beta_hat, ddof=1)
    z_critical = stats.norm.ppf(1 - alpha / 2)

    lower = beta_hat.mean() - z_critical * se / np.sqrt(len(beta_hat)) # noqa
    upper = beta_hat.mean() + z_critical * se / np.sqrt(len(beta_hat)) # noqa

    covers = (beta_true >= lower) & (beta_true <= upper)
    return covers, (lower, upper)


n = 100
N = 10000
beta0_true = 1
beta1_true = 2.5
sigma = 2
X = np.random.uniform(0, 10, n)

beta1_mnk = np.zeros(N)
beta0_mnk = np.zeros(N)
beta1_alt = np.zeros(N)
beta0_alt = np.zeros(N)
beta1_mm = np.zeros(N)
beta0_mm = np.zeros(N)

for i in range(N):
    epsilon = np.random.normal(0, sigma, n)
    Y = beta0_true + beta1_true * X + epsilon

    beta0_mnk[i], beta1_mnk[i] = mnk_estimation(X, Y)
    beta0_alt[i], beta1_alt[i] = alt_estimation(X, Y)
    beta0_mm[i], beta1_mm[i] = mm_estimation(X, Y)

print("\n1. Проверка на несмещённость")
bias_mnk = np.abs(np.mean(beta1_mnk) - beta1_true) # noqa
bias_alt = np.abs(np.mean(beta1_alt) - beta1_true) # noqa
bias_mm = np.abs(np.mean(beta1_mm) - beta1_true) # noqa
print(f"Для МНК: |β1 - 2.5| = {bias_mnk:.9f}")
print(f"Для метода разделения выборки: |β1 - 2.5| = {bias_alt:.9f}")
print(f"Для метода моментов: |β1 - 2.5| = {bias_mm:.9f}")

print("\n2. Проверка на эффективность")
var_mnk = np.var(beta1_mnk, ddof=1)
var_alt = np.var(beta1_alt, ddof=1)
var_mm = np.var(beta1_mm, ddof=1)
print(f"σ² у МНК: {var_mnk:.9f}")
print(f"σ² у метода разделения выборки: {var_alt:.9f}")
print(f"σ² у метода моментов: {var_mm:.9f}")

print("\n3. Среднеквадратическая ошибка (MSE)")
mse_mnk = np.mean((beta1_mnk - beta1_true) ** 2)
mse_alt = np.mean((beta1_alt - beta1_true) ** 2)
mse_mm = np.mean((beta1_mm - beta1_true) ** 2)
print(f"MSE у МНК: {mse_mnk:.9f}")
print(f"MSE у метода разделения выборки: {mse_alt:.9f}")
print(f"MSE у метода моментов: {mse_mm:.9f}")

print("\n4. Относительная эффективность")
eff_alt = var_mnk / var_alt # noqa
eff_mm = var_mnk / var_mm # noqa
print(f"Эффективность оценки метода разделения выборки относительно МНК: {eff_alt:.9f}")
print(f"Эффективность метода моментов относительно МНК: {eff_mm:.9f}")

print("\n5. Доверительные интервалы")

cover_mnk, ci_mnk = coverage_probability(beta1_mnk, beta1_true)
cover_alt, ci_alt = coverage_probability(beta1_alt, beta1_true)
cover_mm, ci_mm = coverage_probability(beta1_mm, beta1_true)

print(f"95% доверительный интервал для МНК: [{ci_mnk[0]:.6f}, {ci_mnk[1]:.6f}]; покрывает: {cover_mnk}")
print(f"95% доверительный интервал для метода разделения выборки: [{ci_alt[0]:.6f}, {ci_alt[1]:.6f}]; покрывает: {cover_alt}")
print(f"95% доверительный интервал для метода моментов: [{ci_mm[0]:.6f}, {ci_mm[1]:.6f}]; покрывает: {cover_mm}")

all_beta1 = np.concatenate([beta1_mnk, beta1_alt, beta1_mm])
x_min = all_beta1.min()
x_max = all_beta1.max()

y_max = 0
for data in [beta1_mnk, beta1_alt, beta1_mm]:
    hist, bins = np.histogram(data, bins=50, density=True)
    y_max = max(y_max, hist.max())
y_max *= 1.05

plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.hist(beta1_mnk, bins=50, label='МНК', density=True)
plt.axvline(beta1_true, color='red', linestyle='--', label='Истинное значение параметра')
plt.xlim(x_min, x_max)
plt.ylim(0, y_max)
plt.legend()
plt.title('Гистограмма распределения оценок (МНК)')
plt.grid(True, alpha=0.25)

plt.subplot(232)
plt.hist(beta1_alt, bins=50, label='Метод разделения выборки', density=True)
plt.axvline(beta1_true, color='red', linestyle='--', label='Истинное значение параметра')
plt.xlim(x_min, x_max)
plt.ylim(0, y_max)
plt.legend()
plt.title('Гистограмма распределения оценок (МРВ)')
plt.grid(True, alpha=0.25)

plt.subplot(233)
plt.hist(beta1_mm, bins=50, label='Метод моментов', density=True)
plt.axvline(beta1_true, color='red', linestyle='--', label='Истинное значение параметра')
plt.xlim(x_min, x_max)
plt.ylim(0, y_max)
plt.legend()
plt.title('Гистограмма распределения оценок (ММ)')
plt.grid(True, alpha=0.25)

plt.subplot(234)
plt.boxplot([beta1_mnk, beta1_alt, beta1_mm], tick_labels=['МНК', 'МРВ', 'ММ'])
plt.axhline(beta1_true, color='red', linestyle='--', label='Истинное значение параметра')
plt.legend()
plt.title('Box-plot для сравнения разброса оценок')
plt.grid(True, alpha=0.25)

n_values = [50, 100, 200, 500]
mse_mnk_by_n = []
mse_alt_by_n = []
mse_mm_by_n = []

for n_value in n_values:
    X_temp_mse = np.random.uniform(0, 10, n_value)
    beta1_mnk_temp_n = []
    beta1_alt_temp_n = []
    beta1_mm_temp_n = []

    for i in range(N):
        epsilon = np.random.normal(0, sigma, n_value)
        Y_temp_n = beta0_true + beta1_true * X_temp_mse + epsilon

        _, beta1_mnk_hat_temp_n = mnk_estimation(X_temp_mse, Y_temp_n)
        _, beta1_alt_hat_temp_n = alt_estimation(X_temp_mse, Y_temp_n)
        _, beta1_mm_hat_temp_n = mm_estimation(X_temp_mse, Y_temp_n)
        beta1_mnk_temp_n.append(beta1_mnk_hat_temp_n)
        beta1_alt_temp_n.append(beta1_alt_hat_temp_n)
        beta1_mm_temp_n.append(beta1_mm_hat_temp_n)

    mse_mnk_by_n.append(np.mean((np.array(beta1_mnk_temp_n) - beta1_true) ** 2))
    mse_alt_by_n.append(np.mean((np.array(beta1_alt_temp_n) - beta1_true) ** 2))
    mse_mm_by_n.append(np.mean((np.array(beta1_mm_temp_n) - beta1_true) ** 2))

plt.subplot(235)
plt.plot(n_values, mse_mnk_by_n, marker='o', label='МНК')
plt.plot(n_values, mse_alt_by_n, marker='o', label='Метод разделения выборки')
plt.plot(n_values, mse_mm_by_n, marker='o', label='Метод моментов')
plt.xlabel('Объём выборки')
plt.ylabel('MSE')
plt.legend()
plt.title('Зависимость MSE от объёма выборки')
plt.xscale('log')
plt.yscale('log')
plt.xticks(n_values, n_values)
plt.grid(True, alpha=0.25)

plt.subplot(236)
probplot(beta1_mnk, dist="norm", plot=plt)
plt.title('Q-Q plot для оценок МНК')
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()

print("\nДополнительные исследования")

print("\n1. Влияние объёма выборки")
print("MSE для разных объёмов выборки:")
for i, n_value in enumerate(n_values):
    print(f"n = {n_value}: MSE у МНК = {mse_mnk_by_n[i]:.9f}, MSE у МРВ = {mse_alt_by_n[i]:.9f}, MSE у ММ = {mse_mm_by_n[i]:.9f}")

print("\n2. Влияние дисперсии ошибок")
sigma_values = [1, 2, 3]
mse_mnk_by_sigma = []
mse_alt_by_sigma = []
mse_mm_by_sigma = []

for sigma in sigma_values:
    beta1_mnk_temp_sigma = []
    beta1_alt_temp_sigma = []
    beta1_mm_temp_sigma = []

    for i in range(N):
        epsilon = np.random.normal(0, sigma, n)
        Y_temp_sigma = beta0_true + beta1_true * X + epsilon

        _, beta1_mnk_hat_temp_sigma = mnk_estimation(X, Y_temp_sigma)
        _, beta1_alt_hat_temp_sigma = alt_estimation(X, Y_temp_sigma)
        _, beta1_mm_hat_temp_sigma = mm_estimation(X, Y_temp_sigma)
        beta1_mnk_temp_sigma.append(beta1_mnk_hat_temp_sigma)
        beta1_alt_temp_sigma.append(beta1_alt_hat_temp_sigma)
        beta1_mm_temp_sigma.append(beta1_mm_hat_temp_sigma)

    mse_mnk_by_sigma.append(np.mean((np.array(beta1_mnk_temp_sigma) - beta1_true) ** 2))
    mse_alt_by_sigma.append(np.mean((np.array(beta1_alt_temp_sigma) - beta1_true) ** 2))
    mse_mm_by_sigma.append(np.mean((np.array(beta1_mm_temp_sigma) - beta1_true) ** 2))

print("MSE для разных дисперсий ошибок:")
for i, sigma_val in enumerate(sigma_values):
    print(f"σ = {sigma_val} (σ² = {sigma_val ** 2}): MSE у МНК = {mse_mnk_by_sigma[i]:.9f}, MSE у МРВ = {mse_alt_by_sigma[i]:.9f}, MSE у ММ = {mse_mm_by_sigma[i]:.9f}")

print("\n3. Проверка робастности (устойчивости к выбросам)")
robustness_mnk_by_outliers = []
robustness_alt_by_outliers = []
robustness_mm_by_outliers = []
outlier_percents = [0, 0.05, 0.1]

for outlier_percent in outlier_percents:
    beta1_mnk_temp_outliers = []
    beta1_alt_temp_outliers = []
    beta1_mm_temp_outliers = []

    for i in range(N):
        epsilon = np.random.normal(0, sigma, n)
        Y_temp_outliers = beta0_true + beta1_true * X + epsilon

        n_outliers = int(n * outlier_percent)
        if n_outliers > 0:
            outlier_indices = np.random.choice(n, n_outliers, replace=False)
            outlier_delta = np.random.normal(0, 5, n_outliers)
            Y_temp_outliers[outlier_indices] += outlier_delta

        _, beta1_mnk_hat_temp_outliers = mnk_estimation(X, Y_temp_outliers)
        _, beta1_alt_hat_temp_outliers = alt_estimation(X, Y_temp_outliers)
        _, beta1_mm_hat_temp_outliers = mm_estimation(X, Y_temp_outliers)
        beta1_mnk_temp_outliers.append(beta1_mnk_hat_temp_outliers)
        beta1_alt_temp_outliers.append(beta1_alt_hat_temp_outliers)
        beta1_mm_temp_outliers.append(beta1_mm_hat_temp_outliers)

    robustness_mnk_by_outliers.append(np.mean((np.array(beta1_mnk_temp_outliers) - beta1_true) ** 2))
    robustness_alt_by_outliers.append(np.mean((np.array(beta1_alt_temp_outliers) - beta1_true) ** 2))
    robustness_mm_by_outliers.append(np.mean((np.array(beta1_mm_temp_outliers) - beta1_true) ** 2))

print("MSE при наличии выбросов:")
for i, percent in enumerate(outlier_percents):
    print(f"{percent * 100}% выбросов: MSE у МНК = {robustness_mnk_by_outliers[i]:.9f}, MSE у МРВ = {robustness_alt_by_outliers[i]:.9f}, MSE у ММ = {robustness_mm_by_outliers[i]:.9f}")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(sigma_values, mse_mnk_by_sigma, marker='o', label='МНК')
plt.plot(sigma_values, mse_alt_by_sigma, marker='o', label='Метод разделения выборки')
plt.plot(sigma_values, mse_mm_by_sigma, marker='o', label='Метод моментов')
plt.xlabel('σ (Cтандартное отклонение)')
plt.ylabel('MSE')
plt.legend()
plt.title('Влияние дисперсии ошибок на MSE')
plt.xscale('log')
plt.yscale('log')

plt.subplot(122)
plt.plot([p * 100 for p in outlier_percents], robustness_mnk_by_outliers, marker='o', label='МНК')
plt.plot([p * 100 for p in outlier_percents], robustness_alt_by_outliers, marker='o', label='Метод разделения выборки')
plt.plot([p * 100 for p in outlier_percents], robustness_mm_by_outliers, marker='o', label='Метод моментов')
plt.xlabel('Процент выбросов')
plt.ylabel('MSE')
plt.legend()
plt.title('Устойчивость методов к выбросам (робастность)')
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()