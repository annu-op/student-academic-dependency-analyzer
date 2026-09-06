import numpy as np
import sys


# ============================================================
# STUDENT ACADEMIC DEPENDENCY ANALYZER
# Linear Algebra & Vector Calculus
# ============================================================

print("=" * 70)
print("        STUDENT ACADEMIC DEPENDENCY ANALYZER")
print("        Linear Algebra & Vector Calculus")
print("=" * 70)


# ------------------------------------------------------------
# 1. USER INPUT
# ------------------------------------------------------------

# Make the script friendly to non-interactive runs (CI / automated)
interactive = sys.stdin is not None and sys.stdin.isatty()

if not interactive:
    # Non-interactive default data
    print("\nNon-interactive mode detected: using sample data.")
    student_name = "Sample Student"
    n = 5
    subjects = ["Mathematics", "Physics", "Chemistry", "English", "History"]
    marks = np.array([85.0, 78.0, 92.0, 74.0, 68.0], dtype=float)

else:

    student_name = input("\nEnter Student Name: ")

    # Number of academic components
    while True:
        try:
            n = int(input("Enter number of academic components (5-8): "))

            if 5 <= n <= 8:
                break
            else:
                print("Please enter a number between 5 and 8.")

        except ValueError:
            print("Please enter a valid number.")


    subjects = []
    marks = []


    print("\nEnter Academic Components and Marks")
    print("-" * 45)

    for i in range(n):

        subject = input(f"Enter name of component {i + 1}: ")

        while True:
            try:
                mark = float(input(f"Enter marks for {subject} (0-100): "))

                if 0 <= mark <= 100:
                    break
                else:
                    print("Marks must be between 0 and 100.")

            except ValueError:
                print("Please enter a valid number.")

        subjects.append(subject)
        marks.append(mark)


    # Convert marks into NumPy array
    marks = np.array(marks, dtype=float)


# ------------------------------------------------------------
# 2. DISPLAY ACADEMIC DATA
# ------------------------------------------------------------

print("\n\n1. ACADEMIC PERFORMANCE DATA")
print("-" * 50)

print(f"Student Name: {student_name}\n")

for subject, mark in zip(subjects, marks):
    print(f"{subject:<30} : {mark:.2f}")


# ------------------------------------------------------------
# 3. CALCULATE AVERAGE
# ------------------------------------------------------------

average = np.mean(marks)

print("\nAverage Marks:", round(average, 2))


if average >= 85:
    performance = "Excellent"
elif average >= 75:
    performance = "Very Good"
elif average >= 60:
    performance = "Good"
elif average >= 50:
    performance = "Average"
else:
    performance = "Needs Improvement"


print("Overall Performance:", performance)


# ------------------------------------------------------------
# 4. CREATE FEATURE MATRIX
# ------------------------------------------------------------

deviation = marks - average

# Safe normalization (guard against division by zero)
denom = np.max(marks) if marks.size > 0 else 0.0
if denom != 0:
    normalized_marks = marks / denom
else:
    normalized_marks = np.zeros_like(marks)

feature_matrix = np.column_stack(
    (marks, deviation, normalized_marks)
)


print("\n\n2. FEATURE MATRIX")
print("-" * 50)

print("Columns:")
print("1. Original Marks")
print("2. Deviation from Average")
print("3. Normalized Marks\n")

print(np.round(feature_matrix, 3))


# ------------------------------------------------------------
# 5. MATRIX RANK
# ------------------------------------------------------------

rank = np.linalg.matrix_rank(feature_matrix)

print("\n\n3. MATRIX RANK")
print("-" * 50)

print("Rank of Feature Matrix:", rank)


# ------------------------------------------------------------
# 6. LINEAR DEPENDENCE / INDEPENDENCE
# ------------------------------------------------------------

number_of_columns = feature_matrix.shape[1]

print("\n\n4. LINEAR DEPENDENCE / INDEPENDENCE")
print("-" * 50)

if rank < number_of_columns:
    print("The feature vectors are LINEARLY DEPENDENT.")
    print("Some academic performance features contain redundant information.")

else:
    print("The feature vectors are LINEARLY INDEPENDENT.")
    print("The academic performance features provide independent directions.")


# ------------------------------------------------------------
# 7. BASIS
# ------------------------------------------------------------

# QR decomposition
Q, R = np.linalg.qr(feature_matrix)

independent_columns = []

# Stable tolerance relative to R's magnitude
absRmax = np.max(np.abs(R)) if R.size > 0 else 0.0
eps = np.finfo(R.dtype).eps if np.issubdtype(R.dtype, np.floating) else 1e-12
tolerance = eps * max(R.shape) * (absRmax if absRmax != 0 else 1.0)

for i in range(min(R.shape)):

    if abs(R[i, i]) > tolerance:
        independent_columns.append(i)


basis = feature_matrix[:, independent_columns] if len(independent_columns) > 0 else np.empty((feature_matrix.shape[0], 0))


print("\n\n5. BASIS OF PERFORMANCE SPACE")
print("-" * 50)

print("Independent column numbers:", independent_columns)

print("\nBasis Matrix:")

print(np.round(basis, 3))


# ------------------------------------------------------------
# 8. GRAM-SCHMIDT ORTHONOGONALIZATION
# ------------------------------------------------------------

def gram_schmidt(matrix):

    # Handle empty input
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))

    orthogonal_vectors = []

    for i in range(matrix.shape[1]):

        vector = matrix[:, i].copy()

        for u in orthogonal_vectors:

            denom = np.dot(u, u)
            if denom == 0:
                continue

            projection = (
                np.dot(vector, u) /
                denom
            ) * u

            vector = vector - projection

        if np.linalg.norm(vector) > 1e-12:
            orthogonal_vectors.append(vector)

    # Normalize vectors
    if len(orthogonal_vectors) == 0:
        return np.empty((matrix.shape[0], 0))

    orthonormal_vectors = []

    for vector in orthogonal_vectors:

        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            unit_vector = vector / norm
            orthonormal_vectors.append(unit_vector)

    if len(orthonormal_vectors) == 0:
        return np.empty((matrix.shape[0], 0))

    return np.column_stack(orthonormal_vectors)


orthonormal_basis = gram_schmidt(basis)


print("\n\n6. GRAM-SCHMIDT ORTHONORMAL BASIS")
print("-" * 50)

# Print safely when empty
if orthonormal_basis.size == 0:
    print("(No orthonormal basis could be formed from the basis matrix.)")
else:
    print(np.round(orthonormal_basis, 3))


# ------------------------------------------------------------
# 9. ORTHOGONALITY VERIFICATION
# ------------------------------------------------------------

print("\n\n7. ORTHOGONALITY VERIFICATION")
print("-" * 50)

if orthonormal_basis.size == 0:
    print("No orthonormal basis available to verify orthogonality.")
else:
    gram_matrix = np.dot(
        orthonormal_basis.T,
        orthonormal_basis
    )

    print("QᵀQ =")
    print(np.round(gram_matrix, 3))

    print("\nIf QᵀQ is approximately the Identity Matrix,")
    print("the basis vectors are orthonormal.")


# ------------------------------------------------------------
# 10. VECTOR PROJECTION
# ------------------------------------------------------------

def project_vector(vector, basis):

    projection = np.zeros_like(vector)

    if basis.size == 0 or basis.shape[1] == 0:
        return projection

    for i in range(basis.shape[1]):

        u = basis[:, i]
        denom = np.dot(u, u)
        if denom == 0:
            continue

        projection += (
            np.dot(vector, u) /
            denom
        ) * u

    return projection


projected_vector = project_vector(
    marks,
    orthonormal_basis
)


print("\n\n8. PROJECTED PERFORMANCE VECTOR")
print("-" * 50)

print(f"{'Component':<30} {'Original':>10} {'Projected':>12}")

print("-" * 55)

for subject, original, projected in zip(
    subjects,
    marks,
    projected_vector
):

    print(
        f"{subject:<30} "
        f"{original:>10.2f} "
        f"{projected:>12.2f}"
    )


# ------------------------------------------------------------
# 11. PROJECTION ERROR
# ------------------------------------------------------------

error = np.linalg.norm(
    marks - projected_vector
)

print("\nProjection Error:", round(error, 4))


# ------------------------------------------------------------
# 12. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n\n9. FINAL PERFORMANCE INTERPRETATION")
print("=" * 60)

print("Student Name       :", student_name)
print("Number of Subjects :", n)
print("Average Marks      :", round(average, 2))
print("Performance Level   :", performance)
print("Matrix Rank        :", rank)
print("Basis Dimension    :", basis.shape[1])

print("\nInterpretation:")

if rank < number_of_columns:

    print(
        "The academic features are linearly dependent."
    )

    print(
        "This indicates that some performance measurements "
        "contain overlapping information."
    )

else:

    print(
        "The academic features are linearly independent."
    )

    print(
        "Each feature contributes a distinct performance direction."
    )


print(
    "\nGram-Schmidt was used to obtain orthonormal "
    "performance directions."
)

print(
    "Vector projection represents the student's academic "
    "performance in the independent performance space."
)


print("\n" + "=" * 70)
print("              PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)
