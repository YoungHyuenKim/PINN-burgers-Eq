import os
import numpy as np
from pyDOE import lhs

if __name__ == "__main__":
    # Burger's Equation
    # u(t,x)
    # f := u_t+ u * u_x -(0.01/pi)u_xx ==0
    #
    # Burgers' Equation :
    # u_t+ u * u_x -(0.01/pi)u_xx ==0
    # condition:
    # u(0,x)= -sin(pi x) << init
    # u(t,-1) = u(t,1) =0  << boundary

    lb = np.array([0, -1])  # t, x
    ub = np.array([1, 1])  # t, x

    N_0 = 100  # initial
    N_d = 100  # boundary
    N_f = 20000  # arb

    # u(0,x)= -sin(pi x)
    bd_1_t = np.zeros((N_0, 1))
    bd_1_x = 2 * np.random.rand(N_0, 1) - 1  # -1~1
    bd_1_u = -np.sin(np.pi * bd_1_x)
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/init.npz"):
        np.savez("data/init", t=bd_1_t, x=bd_1_x, u=bd_1_u)
    else:
        print("data/init.npz exists. do not save")
    # Boundary
    # u(t,-1) = u(t,1) =0
    bd_2_t = np.random.rand(N_d, 1)  # [0, 1]
    bd_2_x = np.where(np.random.rand(N_d, 1) > 0.5, 1.0, -1.0)  # 1 or -1
    bd_2_u = 0 * bd_2_t
    if not os.path.exists("data/boundary.npz"):
        np.savez("data/boundary", t=bd_2_t, x=bd_2_x, u=bd_2_u)
    else:
        print("data/boundary.npz exists. do not save")
    # print(bd_1_x.shape, bd_1_t.shape)

    # 내부 임의 데이터.
    X_f = lb + (ub - lb) * lhs(2, N_f)
    bd_3_t, bd_3_x = X_f[:, 0:1], X_f[:, 1:]
    if not os.path.exists("data/interior.npz"):
        np.savez("data/interior", t=bd_3_t, x=bd_3_x)
    else:
        print("data/interior.npz exists. do not save")

