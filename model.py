import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate



def neural_net_U():
    t_input = Input(shape=1, name="t_input")
    x_input = Input(shape=1, name="x_input")

    x = concatenate([t_input, x_input], axis=1)
    for i in range(9):
        x = Dense(10, activation="tanh", name=f"dense_{i}")(x)
    u = Dense(1, activation="tanh", name="output")(x)

    model = Model(
        inputs=[t_input, x_input],
        outputs=[u],
        name="Burger_equation_Model"
    )
    return model


if __name__ == "__main__":
    model = neural_net_U()
    model.summary()
    print(model.inputs)
    # tf.keras.utils.plot_model(model, "model_plot.png")