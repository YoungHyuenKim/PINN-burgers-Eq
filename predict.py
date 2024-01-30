import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from model import neural_net_U

if __name__ == "__main__":
    model = tf.keras.models.load_model("weight/save_10000.hdf5")
    # layers = [2, 10, 10, 10, 10, 1]
    # model = neural_net_U()

    lb = np.array([0, -1])  # t, x
    ub = np.array([1, 1])  # t, x

    t = np.arange(lb[0], ub[0], 0.01)
    x = np.arange(lb[1], ub[1], 0.01)

    T, X = np.meshgrid(t, x)
    save_shape = X.shape

    TT = T.reshape((-1))
    XX = X.reshape((-1))

    UU = model(inputs=[TT, XX])
    UU = UU.numpy()

    plot_T = TT.reshape(save_shape)
    plot_X = XX.reshape(save_shape)
    plot_U = UU.reshape(save_shape)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot a 3D surface
    ax.plot_surface(plot_T, plot_X, plot_U)

    plt.show()
