import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def func(x):
    return 2 * x**3 + (x - 5) ** 2 + 3 * x**4


x_vals = np.linspace(-5, 5, 400)
y_vals = func(x_vals)
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label=r"$2x^3 + (x-5)^2 + 3x^4$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

plt.savefig("result.png")

result = minimize(func, x0=0)
print(result.x[0], result.fun)
