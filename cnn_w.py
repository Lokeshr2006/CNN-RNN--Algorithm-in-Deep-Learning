import numpy as np

np.random.seed(100)

# STEP 1: Training data
with open("training_data.txt", "r", encoding="utf-8") as file:
    sentences = [line.strip().split() for line in file if line.strip()]

# Vocabulary
vocab = list(set(word for sentence in sentences for word in sentence))
word_to_index = {word: i + 1 for i, word in enumerate(vocab)}
index_to_word = {i + 1: word for i, word in enumerate(vocab)}

PAD_INDEX = 0
vocab_size = len(vocab) + 1

# STEP 2: Hyperparameters
embedding_dim = 8
max_sequence_length = max(len(sentence) for sentence in sentences)

total_filter = 4
filter_size = 3
pool_size = 2
pool_stride = 2

hidden_neurons = 32
output_classes = vocab_size

learning_rate = 0.01
epochs = 500

# Fixed word embeddings (not trainable in this version)
embeddings = np.random.uniform(
    -0.5, 0.5, size=(vocab_size, embedding_dim)
)
embeddings[PAD_INDEX] = 0.0

# STEP 3: Create next-word training examples
training_examples = []

for sentence in sentences:
    for position in range(1, len(sentence)):
        input_words = sentence[:position]
        target_word = sentence[position]

        training_examples.append(
            (input_words, word_to_index[target_word])
        )


def sentence_to_matrix(words):
    # Keep the most recent words if the input is too long
    words = words[-max_sequence_length:]

    # Convert words to IDs and pad on the right
    token_ids = [word_to_index.get(word, PAD_INDEX) for word in words]
    token_ids += [PAD_INDEX] * (max_sequence_length - len(token_ids))

    # Shape: (sequence_length, embedding_dim)
    return embeddings[token_ids]


# STEP 4: Initialize CNN filters and biases
filters = np.random.uniform(
    -0.5, 0.5,
    size=(total_filter, filter_size, filter_size)
)
filter_biases = np.zeros(total_filter)


def conv_forward(image, filters, biases, stride=1):
    num_filters, filter_rows, filter_cols = filters.shape
    total_rows, total_cols = image.shape

    output_rows = (total_rows - filter_rows) // stride + 1
    output_cols = (total_cols - filter_cols) // stride + 1

    feature_maps = np.zeros(
        (num_filters, output_rows, output_cols)
    )

    for f in range(num_filters):
        for i in range(output_rows):
            for j in range(output_cols):
                row_start = i * stride
                col_start = j * stride

                patch = image[
                    row_start:row_start + filter_rows,
                    col_start:col_start + filter_cols
                ]

                feature_maps[f, i, j] = (
                    np.sum(patch * filters[f]) + biases[f]
                )

    return feature_maps


def relu(feature_maps):
    return np.maximum(0, feature_maps)


def maxpool_forward(feature_maps, pool_size=2, stride=2):
    num_maps, total_rows, total_cols = feature_maps.shape

    output_rows = (total_rows - pool_size) // stride + 1
    output_cols = (total_cols - pool_size) // stride + 1

    pooled = np.zeros((num_maps, output_rows, output_cols))
    max_positions = {}

    for m in range(num_maps):
        for i in range(output_rows):
            for j in range(output_cols):
                row_start = i * stride
                col_start = j * stride

                window = feature_maps[
                    m,
                    row_start:row_start + pool_size,
                    col_start:col_start + pool_size
                ]

                pooled[m, i, j] = np.max(window)

                local_pos = np.unravel_index(
                    np.argmax(window), window.shape
                )

                max_positions[(m, i, j)] = (
                    row_start + local_pos[0],
                    col_start + local_pos[1]
                )

    return pooled, max_positions


def softmax(scores):
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def cross_entropy_loss(probabilities, correct_word):
    return -np.log(probabilities[correct_word] + 1e-12)


# Calculate the shape automatically
example_input = np.zeros(
    (max_sequence_length, embedding_dim)
)

example_conv = conv_forward(
    example_input, filters, filter_biases
)
example_relu = relu(example_conv)
example_pool, _ = maxpool_forward(
    example_relu, pool_size, pool_stride
)

flatten_size = example_pool.size

# STEP 5: Initialize fully connected layers
hidden_weights = np.random.uniform(
    -0.5, 0.5, size=(hidden_neurons, flatten_size)
)
hidden_biases = np.zeros(hidden_neurons)

output_weights = np.random.uniform(
    -0.5, 0.5, size=(output_classes, hidden_neurons)
)
output_biases = np.zeros(output_classes)


# STEP 6: Forward pass
def forward_pass(image):
    feature_maps = conv_forward(
        image, filters, filter_biases
    )

    relu_maps = relu(feature_maps)

    pooled_maps, max_positions = maxpool_forward(
        relu_maps, pool_size, pool_stride
    )

    flattened = pooled_maps.flatten()

    hidden_output = hidden_weights @ flattened + hidden_biases
    hidden_relu = relu(hidden_output)

    output_raw = output_weights @ hidden_relu + output_biases
    probabilities = softmax(output_raw)

    cache = {
        "image": image,
        "feature_maps": feature_maps,
        "relu_maps": relu_maps,
        "pooled_maps": pooled_maps,
        "max_positions": max_positions,
        "flattened": flattened,
        "hidden_output": hidden_output,
        "hidden_relu": hidden_relu,
        "probabilities": probabilities,
    }

    return probabilities, cache


# STEP 7: Backward pass pieces
def cal_error_output(probabilities, correct_word):
    error_output = probabilities.copy()
    error_output[correct_word] -= 1
    return error_output


def cal_error_hidden(error_output, hidden_output):
    return (
        output_weights.T @ error_output
    ) * (hidden_output > 0)


def cal_error_flattened(error_hidden):
    return hidden_weights.T @ error_hidden


def cal_error_pooled(error_flattened, pooled_shape):
    return error_flattened.reshape(pooled_shape)


def cal_error_relu_maps(
    error_pooled, max_positions, relu_maps_shape
):
    error_relu_maps = np.zeros(relu_maps_shape)
    num_maps, pooled_rows, pooled_cols = error_pooled.shape

    for m in range(num_maps):
        for i in range(pooled_rows):
            for j in range(pooled_cols):
                orig_row, orig_col = max_positions[(m, i, j)]

                error_relu_maps[m, orig_row, orig_col] += (
                    error_pooled[m, i, j]
                )

    return error_relu_maps


def cal_error_feature_maps(error_relu_maps, relu_maps):
    return error_relu_maps * (relu_maps > 0)


def cal_error_filters(error_feature_maps, image):
    num_filters, filter_rows, filter_cols = filters.shape
    _, output_rows, output_cols = error_feature_maps.shape

    grad_filters = np.zeros_like(filters)
    grad_filter_biases = np.zeros_like(filter_biases)

    for f in range(num_filters):
        grad_filter_biases[f] = np.sum(error_feature_maps[f])

        for i in range(output_rows):
            for j in range(output_cols):
                patch = image[
                    i:i + filter_rows,
                    j:j + filter_cols
                ]

                grad_filters[f] += (
                    error_feature_maps[f, i, j] * patch
                )

    return grad_filters, grad_filter_biases


def backward_pass(cache, correct_word):
    error_output = cal_error_output(
        cache["probabilities"], correct_word
    )

    error_hidden = cal_error_hidden(
        error_output, cache["hidden_output"]
    )

    error_flattened = cal_error_flattened(error_hidden)

    error_pooled = cal_error_pooled(
        error_flattened, cache["pooled_maps"].shape
    )

    error_relu_maps = cal_error_relu_maps(
        error_pooled,
        cache["max_positions"],
        cache["relu_maps"].shape
    )

    error_feature_maps = cal_error_feature_maps(
        error_relu_maps, cache["relu_maps"]
    )

    grad_filters, grad_filter_biases = cal_error_filters(
        error_feature_maps, cache["image"]
    )

    # Gradients for the output layer
    grad_output_weights = np.outer(
        error_output, cache["hidden_relu"]
    )
    grad_output_biases = error_output.copy()

    # Gradients for the hidden layer
    grad_hidden_weights = np.outer(
        error_hidden, cache["flattened"]
    )
    grad_hidden_biases = error_hidden.copy()

    return {
        "grad_filters": grad_filters,
        "grad_filter_biases": grad_filter_biases,
        "grad_hidden_weights": grad_hidden_weights,
        "grad_hidden_biases": grad_hidden_biases,
        "grad_output_weights": grad_output_weights,
        "grad_output_biases": grad_output_biases,
    }


# STEP 8: Update trainable parameters
def update_weights(grads, learning_rate=0.01):
    global filters, filter_biases
    global hidden_weights, hidden_biases
    global output_weights, output_biases

    filters -= learning_rate * grads["grad_filters"]
    filter_biases -= learning_rate * grads["grad_filter_biases"]

    hidden_weights -= learning_rate * grads["grad_hidden_weights"]
    hidden_biases -= learning_rate * grads["grad_hidden_biases"]

    output_weights -= learning_rate * grads["grad_output_weights"]
    output_biases -= learning_rate * grads["grad_output_biases"]


# STEP 9: Train
def train(num_epochs=500, learning_rate=0.01):
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_count = 0

        # Shuffle the examples each epoch
        order = np.random.permutation(len(training_examples))

        for idx in order:
            words, target = training_examples[idx]
            image = sentence_to_matrix(words)

            probabilities, cache = forward_pass(image)

            total_loss += cross_entropy_loss(probabilities, target)

            if np.argmax(probabilities) == target:
                correct_count += 1

            grads = backward_pass(cache, target)
            update_weights(grads, learning_rate)

        avg_loss = total_loss / len(training_examples)
        accuracy = 100 * correct_count / len(training_examples)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"loss: {avg_loss:.4f} - "
                f"training accuracy: {accuracy:.2f}%"
            )


# STEP 10: Predict the next word
def predict_next_word(text):
    words = text.strip().split()

    if not words:
        print("Please enter at least one word.")
        return

    unknown = [word for word in words if word not in word_to_index]

    if unknown:
        print("Unknown word(s):", ", ".join(unknown))
        print("Use words that appear in training_data.txt.")
        return

    image = sentence_to_matrix(words)
    probabilities, _ = forward_pass(image)

    # Exclude the padding token
    probabilities[PAD_INDEX] = 0

    ranked_indices = np.argsort(probabilities)[::-1]

    print("\nTop next-word predictions:")
    for idx in ranked_indices[:5]:
        if idx == PAD_INDEX:
            continue

        print(
            f"{index_to_word[idx]}: "
            f"{probabilities[idx] * 100:.2f}%"
        )


# STEP 11: Show parameter counts
def show_parameter_counts():
    conv_parameters = filters.size + filter_biases.size
    hidden_parameters = hidden_weights.size + hidden_biases.size
    output_parameters = output_weights.size + output_biases.size

    print("\nModel dimensions")
    print("Input embedding matrix:", (max_sequence_length, embedding_dim))
    print("Conv output:", example_conv.shape)
    print("Max-pool output:", example_pool.shape)
    print("Flatten size:", flatten_size)

    print("\nTrainable parameter counts")
    print("Convolution:", conv_parameters)
    print("Hidden layer:", hidden_parameters)
    print("Output layer:", output_parameters)
    print("Total trainable parameters:",
          conv_parameters + hidden_parameters + output_parameters)

    print("Embedding parameters: fixed, not trainable")


if __name__ == "__main__":
    print("Training sentences:", len(sentences))
    print("Vocabulary size including padding:", vocab_size)
    print("Next-word training examples:", len(training_examples))

    show_parameter_counts()

    train(num_epochs=epochs, learning_rate=learning_rate)

    while True:
        user_input = input(
            "\nEnter a sentence to predict the next word (or 'exit'): "
        )

        if user_input.strip().lower() == "exit":
            break

        predict_next_word(user_input)
