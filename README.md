# MNIST-Code

# Install necessary libraries
!pip install tensorflow-federated tensorflow-privacy

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp

# Define a simple CNN model for image classification (e.g., medical image classification)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),  # Example input shape (MNIST-style)
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 output classes
    ])
    return model

# Function to create the TFF model from a Keras model
def model_fn():
    keras_model = create_model()
    return tff.learning.from_keras_model(
        keras_model, 
        input_spec=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
        loss=tf.keras.losses.SparseCategoricalCrossentropy()
    )

# Create a Differentially Private Optimizer
def dp_optimizer():
    return tfp.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=1.0, 
        noise_multiplier=1.1, 
        num_microbatches=256
    )

# Build a Federated Averaging Process
def build_federated_averaging_process(model_fn, client_optimizer_fn, server_optimizer_fn):
    model = model_fn()

    @tff.tf_computation
    def initialize_fn():
        return model_fn().weights

    @tff.tf_computation
    def server_update_fn(model_weights, client_deltas):
        return [w + delta for w, delta in zip(model_weights, client_deltas)]

    @tff.tf_computation
    def client_update_fn(model_weights, dataset):
        model = model_fn()
        model.set_weights(model_weights)
        optimizer = client_optimizer_fn()

        # Define the training loop for a client
        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = model(batch[0], training=True)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return model.get_weights()

    return tff.learning.FederatedLearningProcess(
        initialize_fn=initialize_fn,
        client_update_fn=client_update_fn,
        server_update_fn=server_update_fn
    )

# Now, build the Federated Averaging Process (correct method for TFF >= 0.23)
federated_averaging = build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: dp_optimizer(),  # Use differential privacy optimizer
    server_optimizer_fn=tf.keras.optimizers.SGD
)

# Prepare the dataset (For demonstration, using randomly generated data as a placeholder)
def create_federated_data():
    # Create dummy federated data (replace with real data in practice)
    x = tf.random.normal([10, 28, 28, 1])  # 10 samples per client
    y = tf.random.uniform([10], maxval=10, dtype=tf.int32)  # Random class labels for 10 samples
    return [(x, y)]

# Initialize federated learning process
state = federated_averaging.initialize()

# Simulate federated training for several rounds
for round_num in range(1, 11):  # Run 10 rounds of federated learning
    federated_data = create_federated_data()  # Get data for this round
    state, metrics = federated_averaging.next(state, federated_data)
    print(f"Round {round_num}, Metrics: {metrics}")
