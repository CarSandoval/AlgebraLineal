import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Caso 2: NO diagonalizable (A NO diagonal)
# -----------------------------
A = np.array([[1, 1, 0],
              [0, 1, 1],
              [0, 0, 2]], dtype=float)

rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(400, 3))
Y = (A @ X.T).T

w, V = np.linalg.eig(A)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Caso 2: NO diagonalizable (mg < ma para λ repetido)")

ax.scatter(X[:,0], X[:,1], X[:,2], s=8, alpha=0.35, label="Original")
ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=8, alpha=0.35, label="Transformada A·x")

# Dibujamos direcciones propias numéricas (ojo: para el λ repetido, puede salir 1 sola dirección independiente)
# Para hacerlo más claro, dibujamos el eigenspace de λ=1 resolviendo (A-I)v=0 con SVD
I = np.eye(3)
M = A - 1*I
# vector en el núcleo de M:
_, _, vh = np.linalg.svd(M)
v1 = vh[-1, :]
v1 = v1 / np.linalg.norm(v1)
ax.quiver(0,0,0, v1[0], v1[1], v1[2], length=2.0, normalize=True)
ax.text(1.9*v1[0], 1.9*v1[1], 1.9*v1[2], "eigenspace λ=1 (solo 1 dir.)")

# También dibujamos un eigenvector para λ=2
# (tomamos uno de los que devuelve eig)
idx2 = np.argmin(np.abs(w - 2))
v2 = np.real(V[:, idx2])
v2 = v2 / np.linalg.norm(v2)
ax.quiver(0,0,0, v2[0], v2[1], v2[2], length=2.0, normalize=True)
ax.text(1.9*v2[0], 1.9*v2[1], 1.9*v2[2], "dir. propia λ=2")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()

print("A =\n", A)
print("Autovalores =", w)
