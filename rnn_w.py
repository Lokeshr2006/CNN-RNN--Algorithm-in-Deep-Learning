import numpy as np

np.random.seed(100)

# Hyperparameters
embedding_dim = 8
hidden_state_size = 16
num_hidden_layers = 2
learning_rate = 0.03
epochs = 3000

# Load training data
with open("training_data.txt", "r", encoding="utf-8") as file:
    sentences = [line.strip().split() for line in file if line.strip()]

sentences = [s for s in sentences if len(s) >= 2]

if not sentences:
    raise ValueError("training_data.txt must contain sentences with at least two words.")

# Build vocabulary
words = list(set(word for sentence in sentences for word in sentence))
vocabulary = {word: i for i, word in enumerate(words)}
index_to_word = {i: word for word, i in vocabulary.items()}
V = len(vocabulary)

print("Vocabulary size:", V)

# Initialize parameters
embedding_matrix = np.random.randn(V, embedding_dim) * 0.1

wx = [
    np.random.randn(
        hidden_state_size,
        embedding_dim if layer == 0 else hidden_state_size
    ) * 0.1
    for layer in range(num_hidden_layers)
]

wh = [
    np.random.randn(hidden_state_size, hidden_state_size) * 0.1
    for _ in range(num_hidden_layers)
]

bias = [np.zeros(hidden_state_size) for _ in range(num_hidden_layers)]

wy = np.random.randn(V, hidden_state_size) * 0.1
by = np.zeros(V)


def softmax(scores):
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def forward_pass(tokens):
    T = len(tokens)

    hidden_states = [
        [np.zeros(hidden_state_size) for _ in range(num_hidden_layers)]
        for _ in range(T + 1)
    ]

    layer_inputs = [
        [None for _ in range(num_hidden_layers)]
        for _ in range(T + 1)
    ]

    for t, token in enumerate(tokens, start=1):
        token_index = vocabulary[token]
        x = embedding_matrix[token_index]

        for layer in range(num_hidden_layers):
            layer_inputs[t][layer] = x

            hidden_states[t][layer] = np.tanh(
                wx[layer] @ x
                + wh[layer] @ hidden_states[t - 1][layer]
                + bias[layer]
            )

            x = hidden_states[t][layer]

    final_hidden = hidden_states[T][-1]
    scores = wy @ final_hidden + by
    probabilities = softmax(scores)

    return probabilities, hidden_states, layer_inputs


def cross_entropy_loss(probabilities, target_index):
    return -np.log(probabilities[target_index] + 1e-9)


def backward_pass(tokens, target_index, probabilities, hidden_states, layer_inputs):
    T = len(tokens)

    dwy = np.zeros_like(wy)
    dby = np.zeros_like(by)
    dwx = [np.zeros_like(w) for w in wx]
    dwh = [np.zeros_like(w) for w in wh]
    dbias = [np.zeros_like(b) for b in bias]
    dembedding = np.zeros_like(embedding_matrix)

    dscores = probabilities.copy()
    dscores[target_index] -= 1

    dwy += np.outer(dscores, hidden_states[T][-1])
    dby += dscores

    dh_future = [
        np.zeros(hidden_state_size)
        for _ in range(num_hidden_layers)
    ]

    dh_future[-1] += wy.T @ dscores

    for t in range(T, 0, -1):
        dh_current = [h.copy() for h in dh_future]              

        dh_previous = [
            np.zeros(hidden_state_size)
            for _ in range(num_hidden_layers)
        ]

        for layer in range(num_hidden_layers - 1, -1, -1):
            h = hidden_states[t][layer]
            da = dh_current[layer] * (1 - h ** 2)

            dwx[layer] += np.outer(da, layer_inputs[t][layer])
            dwh[layer] += np.outer(da, hidden_states[t - 1][layer])
            dbias[layer] += da

            dh_previous[layer] += wh[layer].T @ da

            if layer > 0:
                dh_current[layer - 1] += wx[layer].T @ da
            else:
                token_index = vocabulary[tokens[t - 1]]
                dembedding[token_index] += wx[layer].T @ da

        dh_future = dh_previous

    gradients = [dwy, dby]

    for layer in range(num_hidden_layers):
        gradients.extend([dwx[layer], dwh[layer], dbias[layer]])

    gradients.append(dembedding)

    # Gradient clipping
    for gradient in gradients:
        np.clip(gradient, -5, 5, out=gradient)

    return dwy, dby, dwx, dwh, dbias, dembedding


# Create next-word training examples
training_examples = []

for sentence in sentences:
    for i in range(1, len(sentence)):
        training_examples.append((sentence[:i], sentence[i]))

# Train the RNN
for epoch in range(1, epochs + 1):
    total_loss = 0

    for input_tokens, target_word in training_examples:
        target_index = vocabulary[target_word]

        probabilities, hidden_states, layer_inputs = forward_pass(input_tokens)

        total_loss += cross_entropy_loss(probabilities, target_index)

        dwy, dby, dwx, dwh, dbias, dembedding = backward_pass(
            input_tokens,
            target_index,
            probabilities,
            hidden_states,
            layer_inputs
        )

        wy[:] -= learning_rate * dwy
        by[:] -= learning_rate * dby
        embedding_matrix[:] -= learning_rate * dembedding

        for layer in range(num_hidden_layers):
            wx[layer] -= learning_rate * dwx[layer]
            wh[layer] -= learning_rate * dwh[layer]
            bias[layer] -= learning_rate * dbias[layer]

    if epoch % 100 == 0:
        average_loss = total_loss / len(training_examples)
        print(f"Epoch {epoch}/{epochs}, Loss: {average_loss:.4f}")


def predict_next_word(text):
    tokens = text.strip().split()

    if not tokens:
        print("Please enter at least one word.")
        return

    unknown_words = [word for word in tokens if word not in vocabulary]

    if unknown_words:
        print("Unknown word(s):", ", ".join(unknown_words))
        return

    probabilities, _, _ = forward_pass(tokens)

    # Print probabilities for all vocabulary words
    for i in np.argsort(probabilities)[::-1]:
        print(f"{index_to_word[i]}: {probabilities[i] * 100:.6f}%")

    # Sample the next word
    predicted_index = np.random.choice(
    len(probabilities),
    p=probabilities
    )

    predicted_word = index_to_word[predicted_index]

    print("\nNext word:", predicted_word)
    print(f"Selected word probability: {probabilities[predicted_index] * 100:.6f}%")


# Interactive prediction
print("\nRNN training completed!")
print("Enter a phrase to predict the next word. Type 'exit' to quit.")

while True:
    user_input = input("\nEnter text: ").strip()

    if user_input.lower() == "exit":
        break

    predict_next_word(user_input)