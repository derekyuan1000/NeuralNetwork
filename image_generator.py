import random
import math
from neural_network import Matrix, ActivationFunctions

class ImageGenerator:
    def __init__(self, latent_dim=100, image_width=28, image_height=28, channels=1):
        self.latent_dim = latent_dim
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels

        h1_size = 256
        h2_size = 512
        h3_size = 1024
        output_size = image_width * image_height * channels

        self.W1 = Matrix(h1_size, latent_dim).randomize(-0.1, 0.1)
        self.b1 = Matrix(h1_size, 1).randomize(-0.1, 0.1)

        self.W2 = Matrix(h2_size, h1_size).randomize(-0.1, 0.1)
        self.b2 = Matrix(h2_size, 1).randomize(-0.1, 0.1)

        self.W3 = Matrix(h3_size, h2_size).randomize(-0.1, 0.1)
        self.b3 = Matrix(h3_size, 1).randomize(-0.1, 0.1)

        self.W4 = Matrix(output_size, h3_size).randomize(-0.1, 0.1)
        self.b4 = Matrix(output_size, 1).randomize(-0.1, 0.1)

    def generate(self, latent_vector=None):
        if latent_vector is None:
            latent_vector = Matrix(self.latent_dim, 1)
            for i in range(self.latent_dim):
                latent_vector.data[i][0] = random.uniform(-1, 1)

        h1 = self.W1.dot(latent_vector).add(self.b1)
        h1 = h1.map(ActivationFunctions.leaky_relu)

        h2 = self.W2.dot(h1).add(self.b2)
        h2 = h2.map(ActivationFunctions.leaky_relu)

        h3 = self.W3.dot(h2).add(self.b3)
        h3 = h3.map(ActivationFunctions.leaky_relu)

        output = self.W4.dot(h3).add(self.b4)
        output = output.map(ActivationFunctions.sigmoid)

        image = self._reshape_to_image(output.to_array())
        return image

    def _reshape_to_image(self, flat_array):
        image = []
        idx = 0

        for y in range(self.image_height):
            row = []
            for x in range(self.image_width):
                pixel = []
                for c in range(self.channels):
                    pixel.append(flat_array[idx])
                    idx += 1

                if self.channels == 1:
                    row.append(pixel[0])
                else:
                    row.append(pixel)

            image.append(row)

        return image

    def text_to_latent(self, text_embedding):
        if isinstance(text_embedding, list):
            if len(text_embedding) != self.latent_dim:
                projection = Matrix(self.latent_dim, len(text_embedding)).randomize(-0.1, 0.1)
                text_matrix = Matrix.from_array(text_embedding)
                latent = projection.dot(text_matrix)
                return latent
            else:
                return Matrix.from_array(text_embedding)
        else:
            if text_embedding.rows != self.latent_dim:
                projection = Matrix(self.latent_dim, text_embedding.rows).randomize(-0.1, 0.1)
                latent = projection.dot(text_embedding)
                return latent
            else:
                return text_embedding

    def render_ascii(self, image):
        ascii_chars = ' .:-=+*#%@'

        ascii_art = ''
        for row in image:
            for pixel in row:
                char_idx = min(int(pixel * len(ascii_chars)), len(ascii_chars) - 1)
                ascii_art += ascii_chars[char_idx]
            ascii_art += '\n'

        return ascii_art

    def train_step(self, real_images, discriminator, learning_rate=0.001):
        batch_size = len(real_images)
        fake_images = [self.generate() for _ in range(batch_size)]

        feedback = [discriminator.classify(img) for img in fake_images]

        for i in range(self.b4.rows):
            self.b4.data[i][0] += learning_rate * (sum(feedback) / batch_size)

        return sum(feedback) / batch_size


class SimpleDiscriminator:
    def __init__(self, image_width=28, image_height=28, channels=1):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels

        input_size = image_width * image_height * channels
        h1_size = 512
        h2_size = 256
        output_size = 1

        self.W1 = Matrix(h1_size, input_size).randomize(-0.1, 0.1)
        self.b1 = Matrix(h1_size, 1).randomize(-0.1, 0.1)

        self.W2 = Matrix(h2_size, h1_size).randomize(-0.1, 0.1)
        self.b2 = Matrix(h2_size, 1).randomize(-0.1, 0.1)

        self.W3 = Matrix(output_size, h2_size).randomize(-0.1, 0.1)
        self.b3 = Matrix(output_size, 1).randomize(-0.1, 0.1)

    def classify(self, image):
        flat_input = self._flatten_image(image)
        input_matrix = Matrix.from_array(flat_input)

        h1 = self.W1.dot(input_matrix).add(self.b1)
        h1 = h1.map(ActivationFunctions.leaky_relu)

        h2 = self.W2.dot(h1).add(self.b2)
        h2 = h2.map(ActivationFunctions.leaky_relu)

        output = self.W3.dot(h2).add(self.b3)
        output = output.map(ActivationFunctions.sigmoid)

        return output.data[0][0]

    def _flatten_image(self, image):
        flat = []
        for row in image:
            for pixel in row:
                if isinstance(pixel, list):
                    flat.extend(pixel)
                else:
                    flat.append(pixel)
        return flat

    def train_step(self, real_images, fake_images, learning_rate=0.001):
        real_scores = [self.classify(img) for img in real_images]
        fake_scores = [self.classify(img) for img in fake_images]

        real_loss = sum([(1 - score) for score in real_scores]) / len(real_scores)
        fake_loss = sum([score for score in fake_scores]) / len(fake_scores)
        total_loss = real_loss + fake_loss

        for i in range(self.b3.rows):
            self.b3.data[i][0] += learning_rate * (1 - sum(real_scores) / len(real_scores))
            self.b3.data[i][0] -= learning_rate * (sum(fake_scores) / len(fake_scores))

        return total_loss


class TextToImageGenerator:
    def __init__(self, language_model, image_generator):
        self.language_model = language_model
        self.image_generator = image_generator

        embedding_dim = self.language_model.model.embedding_dim
        hidden_size = self.language_model.model.hidden_size
        latent_dim = self.image_generator.latent_dim

        self.projection = Matrix(latent_dim, hidden_size).randomize(-0.1, 0.1)

    def generate_from_text(self, text):
        encoded_text = self.language_model.tokenizer.encode(text)

        outputs = self.language_model.model.forward(encoded_text)
        text_embedding = self.language_model.model.lstm2.h

        latent = self.projection.dot(text_embedding)

        image = self.image_generator.generate(latent)

        return image
