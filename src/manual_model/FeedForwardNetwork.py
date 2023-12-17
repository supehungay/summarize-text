from library import tf

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        
    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])