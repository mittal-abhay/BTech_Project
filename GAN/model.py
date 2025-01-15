import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class RenewableWGAN:
    def __init__(self, sequence_length, feature_dim, conditional_dim=0, latent_dim=100):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.conditional_dim = conditional_dim
        self.latent_dim = latent_dim
        self.n_critic = 5
        self.gp_weight = 10.0

        # Initialize scaler attribute
        self.scaler = None

        # Build networks
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

    def build_generator(self):
        noise_input = layers.Input(shape=(self.latent_dim,))

        # Handle conditional input
        inputs = [noise_input]
        if self.conditional_dim > 0:
            condition_input = layers.Input(shape=(self.conditional_dim,))
            merged_input = layers.Concatenate()([noise_input, condition_input])
            inputs.append(condition_input)
        else:
            merged_input = noise_input

        # Generator architecture
        x = layers.Dense(128 * self.sequence_length)(merged_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((self.sequence_length, 128))(x)

        # Bidirectional LSTM layer
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Self-attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])

        # Output layer
        output = layers.Dense(self.feature_dim, activation='tanh')(x)

        return models.Model(inputs, output)

    def build_critic(self):
        sequence_input = layers.Input(shape=(self.sequence_length, self.feature_dim))

        # Handle conditional input
        inputs = [sequence_input]
        if self.conditional_dim > 0:
            condition_input = layers.Input(shape=(self.conditional_dim,))
            condition_repeated = layers.RepeatVector(self.sequence_length)(condition_input)
            merged_input = layers.Concatenate(axis=-1)([sequence_input, condition_repeated])
            inputs.append(condition_input)
        else:
            merged_input = sequence_input

        # Critic architecture
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(merged_input)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Self-attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])

        x = layers.Flatten()(x)
        x = layers.Dense(64)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        output = layers.Dense(1)(x)

        return models.Model(inputs, output) 

    def build_critic(self):
        sequence_input = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Handle conditional input
        inputs = [sequence_input]
        if self.conditional_dim > 0:
            condition_input = layers.Input(shape=(self.conditional_dim,))
            condition_repeated = layers.RepeatVector(self.sequence_length)(condition_input)
            merged_input = layers.Concatenate(axis=-1)([sequence_input, condition_repeated])
            inputs.append(condition_input)
        else:
            merged_input = sequence_input

        # Critic architecture
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(merged_input)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Self-attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])

        x = layers.Flatten()(x)
        x = layers.Dense(64)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        output = layers.Dense(1)(x)

        return models.Model(inputs, output)

    def gradient_penalty(self, real_samples, fake_samples, conditional_input=None):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        alpha = tf.tile(alpha, [1, self.sequence_length, self.feature_dim])
        
        interpolated = real_samples + alpha * (fake_samples - real_samples)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            if conditional_input is not None:
                pred = self.critic([interpolated, conditional_input])
            else:
                pred = self.critic(interpolated)

        gradients = tape.gradient(pred, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        return tf.reduce_mean((slopes - 1.0) ** 2)

    @tf.function
    def train_step(self, real_sequences, conditional_input=None):
        batch_size = tf.shape(real_sequences)[0]
        
        # Train critic
        critic_loss = 0
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            
            with tf.GradientTape() as tape:
                if conditional_input is not None:
                    fake_sequences = self.generator([noise, conditional_input])
                    real_output = self.critic([real_sequences, conditional_input])
                    fake_output = self.critic([fake_sequences, conditional_input])
                else:
                    fake_sequences = self.generator(noise)
                    real_output = self.critic(real_sequences)
                    fake_output = self.critic(fake_sequences)
                
                gp = self.gradient_penalty(real_sequences, fake_sequences, conditional_input)
                critic_loss = tf.reduce_mean(fake_output - real_output) + self.gp_weight * gp

            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            if conditional_input is not None:
                fake_sequences = self.generator([noise, conditional_input])
                fake_output = self.critic([fake_sequences, conditional_input])
            else:
                fake_sequences = self.generator(noise)
                fake_output = self.critic(fake_sequences)
            
            generator_loss = -tf.reduce_mean(fake_output)

        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return critic_loss, generator_loss

    def train(self, sequences, epochs, batch_size):
        # Store training losses
        critic_losses = []
        generator_losses = []
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            np.random.shuffle(sequences)
            
            epoch_critic_losses = []
            epoch_generator_losses = []
            
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                if len(batch_sequences) < batch_size:
                    continue
                    
                # Convert to tensor if needed
                batch_sequences = tf.convert_to_tensor(batch_sequences, dtype=tf.float32)
                
                # Train on batch
                critic_loss, generator_loss = self.train_step(batch_sequences)
                
                epoch_critic_losses.append(critic_loss)
                epoch_generator_losses.append(generator_loss)
            
            # Compute epoch losses
            avg_critic_loss = tf.reduce_mean(epoch_critic_losses)
            avg_generator_loss = tf.reduce_mean(epoch_generator_losses)
            
            critic_losses.append(avg_critic_loss)
            generator_losses.append(avg_generator_loss)
            
            print(f"Epoch {epoch + 1}/{epochs}, Critic Loss: {avg_critic_loss:.4f}, Generator Loss: {avg_generator_loss:.4f}")
        
        return critic_losses, generator_losses

    def generate_scenarios(self, num_scenarios, conditional_input=None):
        noise = tf.random.normal([num_scenarios, self.latent_dim])
        if conditional_input is not None:
            scenarios = self.generator.predict([noise, conditional_input])
        else:
            scenarios = self.generator.predict(noise)
        return scenarios