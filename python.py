import numpy as np


def get_grade(m):

    if m >= 90: return "O", 10

    if m >= 80: return "A+", 9

    if m >= 70: return "A", 8

    if m >= 60: return "B+", 7

    if m >= 55: return "B", 6

    if m >= 50: return "C", 5

    if m >= 40: return "P", 4

    return "F", 0


def gram_schmidt(vectors):

    Q = []

    for v in vectors:

        u = v.astype(float).copy()

        for q in Q:

            u -= np.dot(u, q) * q

        n = np.linalg.norm(u)

        if n > 1e-9:

            Q.append(u / n)

    return Q


def analyze(subjects):

    X = np.array([s["marks"] for s in subjects], dtype=float)

    C = np.array([s["credits"] for s in subjects], dtype=float)

    GP = np.array([get_grade(x)[1] for x in X], dtype=float)


    sgpa = np.sum(C * GP) / np.sum(C)

    mean = np.mean(X)

    D = X - mean

    N = X / np.max(X) if np.max(X) else np.zeros_like(X)

    A = np.column_stack((X, D, N))

    r = np.linalg.matrix_rank(A)


    vectors = [X, D, N]

    names = ["Marks Vector X", "Deviation Vector D", "Normalized Vector N"]

    basis, basis_names = [], []

    old_rank = 0

    for v, name in zip(vectors, names):

        test = np.column_stack(basis + [v]) if basis else v.reshape(-1, 1)

        new_rank = np.linalg.matrix_rank(test)

        if new_rank > old_rank:

            basis.append(v)

            basis_names.append(name)

            old_rank = new_rank


    Q = gram_schmidt(basis)

    projection = sum(np.dot(X, q) * q for q in Q) if Q else np.zeros_like(X)


    return sgpa, X, D, N, A, r, basis_names, Q, projection


# Enter any number of subjects

subjects = []

n = int(input("Number of subjects: "))

for i in range(n):

    print("\nSubject", i + 1)

    name = input("Subject Name: ")

    marks = float(input("Marks (0-100): "))

    credits = float(input("Credits: "))

    subjects.append({"name": name, "marks": marks, "credits": credits})


sgpa, X, D, N, A, rank, basis, Q, projection = analyze(subjects)


print("\n--- RESULT ---")

for s in subjects:

    grade, point = get_grade(s["marks"])

    print(f'{s["name"]}: {s["marks"]} | {grade} | Grade Point {point}')

print(f"SGPA: {sgpa:.2f}")

print("Feature Matrix A = [X D N]")

print(np.round(A, 4))

print("Rank:", rank)

print("Dependency:", "Linearly Dependent" if rank < 3 else "Linearly Independent")

print("Basis:", ", ".join(basis))

print("Gram-Schmidt Orthonormal Basis:")

for i, q in enumerate(Q, 1):

    print(f"q{i} =", np.round(q, 4))

print("Projection of X:")

print(np.round(projection, 4))
