import tensorflow as tf


def my_clip_by_global_norm(t_list, clip_norm):
    """
    Clips gradients by global norm.

    The global norm is the square root of the sum of the L2 norm of each gradient.

    If the global norm is greater than `clip_norm`, scales down the gradients
    such that the global norm becomes `clip_norm`.

    Args:
        t_list: list of gradient tensors.
        clip_norm: a float specifying the maximum global norm.

    Returns:
        A list of clipped gradient tensors and the global norm of the input tensors.
    """
    # Calculate the global norm of the gradients.
    global_norm = tf.sqrt(sum([tf.norm(t)**2 for t in t_list]))

    # Calculate the scaling factor for the gradients.
    scale = clip_norm / tf.maximum(global_norm, clip_norm)

    # Apply the scaling factor to the gradients.
    t_list_scaled = [t * scale for t in t_list]

    return t_list_scaled, global_norm


if __name__ == '__main__':

    # Define a simple model
    layer = tf.keras.layers.Dense(2, activation='relu')

    # Input tensor
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)

        # Compute loss value
        loss = tf.reduce_mean(y**2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)

    # Set the clip norm value
    clip_norm = 40

    # Use TensorFlow's clip_by_global_norm function
    tf_grad, tf_norm = tf.clip_by_global_norm(grad, clip_norm)

    # Use the custom my_clip_by_global_norm function
    my_grad, my_norm = my_clip_by_global_norm(grad, clip_norm)

    # Print the results
    print('==== Compare gradients ====')
    print(f'tf_grad: {tf_grad}, my_grad: {my_grad}')
    print()

    print('==== Compare norms ====')
    print(f'tf_norm: {tf_norm}, my_norm: {my_norm}')
