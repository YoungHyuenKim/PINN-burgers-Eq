import tensorflow as tf
import numpy as np

from model import neural_net_U
from pyDOE import lhs

if __name__ == "__main__":

    # layers = [2, 10, 10, 10, 10, 1]

    model = neural_net_U()
    model.summary()
    print(model.inputs)
    tf.keras.utils.plot_model(model, "model_plot.png")
    # prepare data set.

    # Data load
    init_data = np.load("data/init.npz")
    boundary_data = np.load("data/boundary.npz")
    interior_data = np.load("data/interior.npz")

    bd_1_t, bd_1_x, bd_1_u = init_data['t'], init_data['x'], init_data['u']
    bd_2_t, bd_2_x, bd_2_u = boundary_data['t'], boundary_data['x'], boundary_data['u']
    bd_3_t, bd_3_x = interior_data['t'], interior_data['x']

    # Train
    epochs = 10000
    lr = 1e-3
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        with tf.GradientTape() as tape_1:
            t1 = tf.Variable(bd_1_t, dtype=tf.float32)
            x1 = tf.Variable(bd_1_x, dtype=tf.float32)
            pred_u1 = model(inputs=[t1, x1])

            loss_1 = tf.reduce_mean(tf.square(pred_u1 - bd_1_u))
            with tf.GradientTape() as tape_2:
                t2 = tf.Variable(bd_2_t, dtype=tf.float32)
                x2 = tf.Variable(bd_2_x, dtype=tf.float32)
                pred_u1 = model(inputs=[t2, x2])
                loss_2 = tf.reduce_mean(tf.square(pred_u1 - bd_2_u))

            with tf.GradientTape() as tape_3_1:
                with tf.GradientTape() as tape_3_2:
                    t3 = tf.Variable(bd_3_t)
                    x3 = tf.Variable(bd_3_x)
                    pred_u2 = model(inputs=[t3, x3])
                    u_t, u_x = tape_3_1.gradient(pred_u2, [t3, x3])
                u_xx = tape_3_2.gradient(u_x, x3)
                f1 = u_t - (0.01 / np.pi) * u_xx
                f1 = tf.cast(f1, dtype=tf.float32)  # u_t - c * u_xx

                f2 = pred_u2 * tf.cast(u_x, dtype=tf.float32)  # u* u_x
                f = f1 + f2
                loss_3 = tf.reduce_mean(tf.square(f))

            loss = loss_1 + loss_2 + loss_3

        grad = tape_1.gradient(loss, model.variables)

        optimizer.apply_gradients(zip(grad, model.variables))

        print(f"{epoch + 1:04}/{epochs:04}  loss : {loss}")
        if (epoch + 1) % 100 == 0:
            print("save Model ")
            model.save(f"weight/save_{epoch + 1}.hdf5")
