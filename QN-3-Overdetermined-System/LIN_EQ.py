# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# QN 3(I): First visualize the system of linear equations.
# Prepare for visualization
x1 = np.linspace(-5, 5, 1000)
x2_1 = (.97-.25*x1) / 0.03
x2_2 = (0.48+0.37*x1) / 0.17
x2_3 = (2.20-1.17*x1)
x2_4 = (-1.19+1.09*x1) / (-0.17)
x2_5 = (1.73-x1) / 1.19

# Visualize
plt.plot(x1, x2_1)
plt.plot(x1, x2_2)
plt.plot(x1, x2_3)
plt.plot(x1, x2_4)
plt.plot(x1, x2_5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["lin-eq-1", "lin-eq-2", "lin-eq-3", "lin-eq-4", "lin-eq-5"])
print("Saving Image")
plt.savefig("op.jpg", dpi=500)
plt.show()
print("Check op.jpg image file.")

# QN 3(II): (Method-2) We would like to find the best possible solution for this linear system
A = np.array([
    [0.25, 0.03],
    [-0.37, 0.17],
    [1.17, 1],
    [-1.09, -0.17],
    [1, 1.19]
])

B = np.array([[0.97], [0.48], [2.20], [-1.19], [1.73]])

# Compute pseudoinverse of A
A_plus = np.linalg.pinv(A)

# Result
res = A_plus.dot(B)
print("Solution of Overdetermined System", res)
