import tensorflow as tf

class TextCNN(tf.keras.Model):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0.0,
                 filter_sizes=[3, 4, 5, 6], mode=0):
        super(TextCNN, self).__init__()
        self.mode = mode
        self.l2_reg_lambda = l2_reg_lambda
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = 256  # Number of filters per filter size

        # Regularization
        self.l2_regularizer = tf.keras.regularizers.L2(l2_reg_lambda)

        # Embedding layers
        if mode in [4, 5]:
            self.char_embedding = tf.keras.layers.Embedding(
                input_dim=char_ngram_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer='uniform',
                name="char_embedding"
            )

        if mode in [2, 3, 4, 5]:
            self.word_embedding = tf.keras.layers.Embedding(
                input_dim=word_ngram_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer='uniform',
                name="word_embedding"
            )

        if mode in [1, 3, 5]:
            self.char_seq_embedding = tf.keras.layers.Embedding(
                input_dim=char_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer='uniform',
                name="char_seq_embedding"
            )

        # Convolution and pooling layers for word inputs
        if mode in [2, 3, 4, 5]:
            self.conv_layers = []
            for filter_size in filter_sizes:
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_size=(filter_size, embedding_size),
                        activation='relu',
                        kernel_initializer='glorot_uniform',
                        name=f"conv_{filter_size}"
                    )
                )

        # Convolution and pooling layers for character inputs
        if mode in [1, 3, 5]:
            self.char_conv_layers = []
            for filter_size in filter_sizes:
                self.char_conv_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_size=(filter_size, embedding_size),
                        activation='relu',
                        kernel_initializer='glorot_uniform',
                        name=f"char_conv_{filter_size}"
                    )
                )

        # Fully connected layers
        total_filters = self.num_filters * len(filter_sizes)
        if mode in [3, 5]:
            fc_input_dim = total_filters * 2  # Concatenated word and char features
        else:
            fc_input_dim = total_filters

        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.output_layer = tf.keras.layers.Dense(2, kernel_regularizer=self.l2_regularizer)

    def call(self, inputs, training=False, dropout_keep_prob=1.0):
        # Apply embeddings
        pooled_outputs = []
        l2_loss = 0.0

        if self.mode in [4, 5]:
            x_char = inputs['input_x_char']
            x_char_pad_idx = inputs['input_x_char_pad_idx']
            embedded_x_char = self.char_embedding(x_char)
            embedded_x_char = tf.multiply(embedded_x_char, x_char_pad_idx)
            sum_ngram_x_char = tf.reduce_sum(embedded_x_char, axis=2)

        if self.mode in [2, 3, 4, 5]:
            x_word = inputs['input_x_word']
            embedded_x_word = self.word_embedding(x_word)

        if self.mode in [1, 3, 5]:
            x_char_seq = inputs['input_x_char_seq']
            embedded_x_char_seq = self.char_seq_embedding(x_char_seq)

        # Combine embeddings for word-level convolution
        if self.mode in [4, 5]:
            sum_ngram_x = sum_ngram_x_char + embedded_x_word
            x_conv_input = tf.expand_dims(sum_ngram_x, -1)
        elif self.mode in [2, 3]:
            x_conv_input = tf.expand_dims(embedded_x_word, -1)

        # Word-level convolution and pooling
        if self.mode in [2, 3, 4, 5]:
            pooled_outputs = []
            for conv_layer in self.conv_layers:
                conv = conv_layer(x_conv_input)
                pool_size = (conv.shape[1], 1)
                pooled = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='valid')(conv)
                pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, axis=-1)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])
            h_pool_flat = self.dropout(h_pool_flat, training=training and (dropout_keep_prob < 1.0))

        # Character-level convolution and pooling
        if self.mode in [1, 3, 5]:
            char_x_conv_input = tf.expand_dims(embedded_x_char_seq, -1)
            char_pooled_outputs = []
            for conv_layer in self.char_conv_layers:
                conv = conv_layer(char_x_conv_input)
                pool_size = (conv.shape[1], 1)
                pooled = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='valid')(conv)
                char_pooled_outputs.append(pooled)

            h_char_pool = tf.concat(char_pooled_outputs, axis=-1)
            h_char_pool_flat = tf.reshape(h_char_pool, [-1, self.num_filters * len(self.filter_sizes)])
            h_char_pool_flat = self.dropout(h_char_pool_flat, training=training and (dropout_keep_prob < 1.0))

        # Combine word and character features
        if self.mode in [3, 5]:
            conv_output = tf.concat([h_pool_flat, h_char_pool_flat], axis=1)
        elif self.mode in [2, 4]:
            conv_output = h_pool_flat
        elif self.mode == 1:
            conv_output = h_char_pool_flat

        # Fully connected layers
        output = self.fc1(conv_output)
        output = self.fc2(output)
        output = self.fc3(output)
        logits = self.output_layer(output)

        return logits
