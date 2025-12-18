import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Caso 1: diagonalizable (A NO diagonal)
# -----------------------------
P = np.array([[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1]], dtype=float)

D = np.diag([2, 2, 3])  # λ=2 tiene mult. algebraica 2
A = P @ D @ np.linalg.inv(P)  # A no diagonal pero diagonalizable

# Puntos del cubo [-1,1]^3
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(400, 3))
Y = (A @ X.T).T

# Autovalores y autovectores
w, V = np.linalg.eig(A)  # columnas de V son vectores propios

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Caso 1: diagonalizable (mg = ma para λ repetido)")

# nube original y transformada
ax.scatter(X[:,0], X[:,1], X[:,2], s=8, alpha=0.35, label="Original")
ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=8, alpha=0.35, label="Transformada A·x")

# dibuja direcciones propias (normalizadas)
for i in range(3):
    v = np.real(V[:, i])
    v = v / np.linalg.norm(v)
    ax.quiver(0,0,0, v[0], v[1], v[2], length=2.0, normalize=True)
    ax.text(1.9*v[0], 1.9*v[1], 1.9*v[2], f"v{i+1}")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()

print("A =\n", A)
print("Autovalores =", w)
