from torchvision import datasets
import numpy as np

train_data = datasets.MNIST(root='./data', train=True, download=True)
test_data = datasets.MNIST(root='./data', train=False, download=True)

train_images = train_data.data.numpy()
train_labels = train_data.targets.numpy()

test_images = test_data.data.numpy()
test_labels = test_data.targets.numpy()

small_images = train_images
small_labels = train_labels

total_filter = 4
filter_size = 3

filters = np.random.uniform(-0.5, 0.5, size=(total_filter, filter_size, filter_size))
filter_biases = np.zeros(total_filter)


def conv_forward(image, filters, biases, stride=1):
    num_filters, total_filter_row, total_filter_col = filters.shape
    total_rows, total_cols = image.shape
    feature_map_rowsize = (total_rows - total_filter_row) // stride + 1
    feature_map_colsize = (total_cols - total_filter_col) // stride + 1

    feature_maps = np.zeros((num_filters, feature_map_rowsize, feature_map_colsize))

    for f in range(num_filters):
        for i in range(feature_map_rowsize):
            for j in range(feature_map_colsize):
                row_start = i * stride
                col_start = j * stride
                patch = image[row_start:row_start+total_filter_row,
                               col_start:col_start+total_filter_col]
                feature_maps[f, i, j] = np.sum(patch * filters[f]) + biases[f]

    return feature_maps

def relu(feature_maps):
    return np.maximum(0, feature_maps)


def maxpool_forward(feature_maps, pool_size=2, stride=2):
    num_maps, total_rows, total_cols = feature_maps.shape
    pooled_rowsize = (total_rows - pool_size) // stride + 1
    pooled_colsize = (total_cols - pool_size) // stride + 1

    pooled = np.zeros((num_maps, pooled_rowsize, pooled_colsize))
    max_positions = {}

    for m in range(num_maps):
        for i in range(pooled_rowsize):
            for j in range(pooled_colsize):
                row_start = i * stride
                col_start = j * stride
                window = feature_maps[m, row_start:row_start+pool_size,
                                          col_start:col_start+pool_size]
                pooled[m, i, j] = np.max(window)
                local_pos = np.unravel_index(np.argmax(window), window.shape)
                max_positions[(m, i, j)] = (row_start + local_pos[0], col_start + local_pos[1])

    return pooled, max_positions

def softmax(scores):
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)

def cross_entropy_loss(probabilities, correct_digit):
    return -np.log(probabilities[correct_digit] + 1e-12)


flatten_size = total_filter * 13 * 13

hidden_neurons = 32
output_classes = 10

hidden_weights = np.random.uniform(-0.5, 0.5, size=(hidden_neurons, flatten_size))
hidden_biases = np.zeros(hidden_neurons)

output_weights = np.random.uniform(-0.5, 0.5, size=(output_classes, hidden_neurons))
output_biases = np.zeros(output_classes)

def forward_pass(image):
    feature_maps = conv_forward(image, filters, filter_biases, stride=1)
    relu_maps = relu(feature_maps)
    pooled_maps, max_positions = maxpool_forward(relu_maps, pool_size=2, stride=2)

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
        "output_raw": output_raw,
        "probabilities": probabilities,
    }
    return probabilities, cache


# STEP 5: Backward pass pieces


def cal_error_output(probabilities, correct_digit):
    error_output = probabilities.copy()
    error_output[correct_digit] -= 1
    return error_output


def cal_error_hidden(error_output, hidden_output):
    error_hidden = (output_weights.T @ error_output) * (hidden_output > 0)
    return error_hidden


def cal_error_flattened(error_hidden):
    error_flattened = hidden_weights.T @ error_hidden
    return error_flattened


def cal_error_pooled(error_flattened, pooled_maps_shape):
    error_pooled = error_flattened.reshape(pooled_maps_shape)
    return error_pooled


def cal_error_relu_maps(error_pooled, max_positions, relu_maps_shape):
    error_relu_maps = np.zeros(relu_maps_shape)
    num_maps, pooled_rowsize, pooled_colsize = error_pooled.shape

    for m in range(num_maps):
        for i in range(pooled_rowsize):
            for j in range(pooled_colsize):
                orig_row, orig_col = max_positions[(m, i, j)]
                error_relu_maps[m, orig_row, orig_col] += error_pooled[m, i, j]

    return error_relu_maps


def cal_error_feature_maps(error_relu_maps, relu_maps):
    error_feature_maps = error_relu_maps * (relu_maps > 0)
    return error_feature_maps


def cal_error_filters(error_feature_maps, image):
    num_filters, filter_row, filter_col = filters.shape
    _, out_h, out_w = error_feature_maps.shape

    grad_filters = np.zeros_like(filters)
    grad_filter_biases = np.zeros(num_filters)

    for f in range(num_filters):
        grad_filter_biases[f] = np.sum(error_feature_maps[f])
        for i in range(out_h):
            for j in range(out_w):
                patch = image[i:i+filter_row, j:j+filter_col]
                grad_filters[f] += error_feature_maps[f, i, j] * patch

    return grad_filters, grad_filter_biases


def backward_pass(cache, correct_digit):
    error_output = cal_error_output(cache["probabilities"], correct_digit)
    error_hidden = cal_error_hidden(error_output, cache["hidden_output"])
    error_flattened = cal_error_flattened(error_hidden)
    error_pooled = cal_error_pooled(error_flattened, cache["pooled_maps"].shape)
    error_relu_maps = cal_error_relu_maps(error_pooled, cache["max_positions"], cache["relu_maps"].shape)
    error_feature_maps = cal_error_feature_maps(error_relu_maps, cache["relu_maps"])

    grad_filters, grad_filter_biases = cal_error_filters(error_feature_maps, cache["image"])

    # weight gradients for the MLP part
    grad_output_weights = np.outer(error_output, cache["hidden_relu"])
    grad_output_biases = error_output

    grad_hidden_weights = np.outer(error_hidden, cache["flattened"])
    grad_hidden_biases = error_hidden

    grads = {
        "grad_filters": grad_filters,
        "grad_filter_biases": grad_filter_biases,
        "grad_hidden_weights": grad_hidden_weights,
        "grad_hidden_biases": grad_hidden_biases,
        "grad_output_weights": grad_output_weights,
        "grad_output_biases": grad_output_biases,
    }
    return grads


def update_weights(grad_filters, grad_filter_biases, grad_hidden_weights, grad_hidden_biases,
                    grad_output_weights, grad_output_biases, learning_rate=0.01):
    global filters, filter_biases, hidden_weights, hidden_biases, output_weights, output_biases

    filters -= learning_rate * grad_filters
    filter_biases -= learning_rate * grad_filter_biases
    hidden_weights -= learning_rate * grad_hidden_weights
    hidden_biases -= learning_rate * grad_hidden_biases
    output_weights -= learning_rate * grad_output_weights
    output_biases -= learning_rate * grad_output_biases



def train(num_train_images, num_epochs=3, learning_rate=0.01):
    images_to_use = small_images[:num_train_images]
    labels_to_use = small_labels[:num_train_images]

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_count = 0

        for idx in range(len(images_to_use)):
            image = images_to_use[idx].astype(float) / 255.0
            label = labels_to_use[idx]

            probabilities, cache = forward_pass(image)

            loss = cross_entropy_loss(probabilities, label)
            total_loss += loss

            predicted_digit = np.argmax(probabilities)
            if predicted_digit == label:
                correct_count += 1

            grads = backward_pass(cache, label)

            update_weights(
                grads["grad_filters"], grads["grad_filter_biases"],
                grads["grad_hidden_weights"], grads["grad_hidden_biases"],
                grads["grad_output_weights"], grads["grad_output_biases"],
                learning_rate=learning_rate,
            )

        avg_loss = total_loss / len(images_to_use)
        accuracy = correct_count / len(images_to_use) * 100

        print(f"Epoch {epoch+1}/{num_epochs} - avg loss: {avg_loss:.4f} - accuracy: {accuracy:.2f}%")


def test(num_test_images):
    correct_count = 0

    for idx in range(num_test_images):
        image = test_images[idx].astype(float) / 255.0
        actual_label = test_labels[idx]

        probabilities, cache = forward_pass(image)
        guess = np.argmax(probabilities)

        if guess == actual_label:
            correct_count += 1

    print(f"out of {num_test_images} images, {num_test_images} images tested successfully")
    print(f"{correct_count}/{num_test_images} images is correct")

if __name__ == "__main__":
    print(f"Full training set available: {len(small_images)} images")
    num_train_images = int(input("How many images should I train on? "))

    train(num_train_images=num_train_images, num_epochs=3, learning_rate=0.01)

    answer = input("Test the images? (y/n): ")
    if answer.strip().lower() == "y":
        num_test_images = len(test_images) // 2 
        test(num_test_images)
    else:
        print("Skipping testing.")