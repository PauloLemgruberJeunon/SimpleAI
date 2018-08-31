import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from random import uniform


RANGE = (-10, 10)
EPOCHS = 1000


def f(x):
    return (np.power(x, 2)) + 5
    # return 2 * x + 3


def function_draw_array():
    return [(x, f(x)) for x in np.arange(*RANGE, 0.1)]


def generate_dataset():
    train_array = []
    test_array = []
    for i in range(1000):
        curr_train_rand = uniform(*RANGE)
        train_array.append((curr_train_rand, f(curr_train_rand)))

        curr_test_rand = uniform(*RANGE)
        test_array.append((curr_test_rand, f(curr_test_rand)))
    return train_array, test_array


def train(model, train_data):
    x, y = zip(*train_data)
    x = np.array(x)
    y = np.array(y)
    print(model.fit(x, y, epochs=EPOCHS, validation_split=0.2, verbose=2))


def predict(model, test_data):
    x, y = zip(*test_data)
    x = np.array(x)
    y = np.array(y)
    test_predictions = model.predict(x).flatten()
    plt.scatter(x, test_predictions)
    x, y = zip(*function_draw_array())
    plt.scatter(x, y)
    plt.show()


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation=tf.nn.relu, input_shape=(1,)),
        tf.keras.layers.Dense(5, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mse', metrics=['mae'])
    return model


model = create_model()
train_data, test_data = generate_dataset()
train(model, train_data)
predict(model, test_data)
