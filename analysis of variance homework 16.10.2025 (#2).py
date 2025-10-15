import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([0, 20, 40, 60, 80, 100])
x_full = np.repeat(x, 4)

y_data = np.array([
    [727, 884, 1073, 1194, 1350, 1442],
    [721, 880, 1050, 1184, 1291, 1369],
    [743, 885, 1045, 1205, 1291, 1458],
    [746, 890, 1033, 1180, 1323, 1459]])
y_full = y_data.flatten()

def quadratic_regression(x, y):  # noqa
    n = len(x)  # noqa
    X = np.column_stack([np.ones(n), x, x**2])
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    y_predicted = X @ coefficients
    SSE = np.sum((y - y_predicted)**2)  # noqa

    return coefficients, y_predicted, SSE


def linear_regression(x, y):  # noqa
    n = len(x)  # noqa
    X = np.column_stack([np.ones(n), x])
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    y_predicted = X @ coefficients
    SSE = np.sum((y - y_predicted)**2)  # noqa

    return coefficients, y_predicted, SSE

coefficients_quadratic, y_quadratic_predicted, SSE_H0 = quadratic_regression(x_full, y_full)
print(f"\nКвадратичная регрессия:")
print(f"y = {coefficients_quadratic[2]:.6f}x² + {coefficients_quadratic[1]:.4f}x + {coefficients_quadratic[0]:.4f}")

coefficients_linear, y_linear_predicted, SSE_H1 = linear_regression(x_full, y_full)
print(f"\nЛинейная регрессия:")
print(f"y = {coefficients_linear[1]:.4f}x + {coefficients_linear[0]:.4f}")

n = len(y_full)
r = 3
t = 1

SSH = SSE_H1 - SSE_H0  # noqa
SSE = SSE_H0

SSH_normalized = SSH / t
SSE_normalized = SSE / (n - r)

F_statistic = SSH_normalized / SSE_normalized
print(f"\nF-статистика = {F_statistic:.4f}")

alpha = 0.05
F_critical = stats.f.ppf(1 - alpha, t, n - r)

if F_statistic > F_critical:
    print(f"Отвергаем H0: связь линейная.")
else:
    print(f"Нет оснований отвергать H0: связь квадратичная.")

x_plot = np.linspace(0, 100, 100)
y_quad_plot = coefficients_quadratic[0] + coefficients_quadratic[1] * x_plot + coefficients_quadratic[2] * x_plot ** 2
y_lin_plot = coefficients_linear[0] + coefficients_linear[1] * x_plot

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_quad_plot, 'g-', linewidth=2, label='H0: Квадратичная')
plt.plot(x_plot, y_lin_plot, 'r--', linewidth=2, label='H1: Линейная')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графики линейной и квадратичной регрессий')
plt.legend()
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()