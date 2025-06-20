import math
import random
from neural_network import Matrix, ActivationFunctions, Embedding, LSTM

class LanguageModel:
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, temperature=0.8):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.temperature = temperature

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm1 = LSTM(embedding_dim, hidden_size)
        self.lstm2 = LSTM(hidden_size, hidden_size)

        self.W_out = Matrix(vocab_size, hidden_size).randomize(-0.1, 0.1)
        self.b_out = Matrix(vocab_size, 1).randomize(-0.1, 0.1)

        self.learning_rate = 0.01

        self.context_size = 3

    def forward(self, word_indices, reset_state=True):
        if reset_state:
            self.lstm1.reset_state()
            self.lstm2.reset_state()

        outputs = []

        for word_idx in word_indices:
            embedded = self.embedding.forward(word_idx)

            hidden1 = self.lstm1.forward(embedded)
            hidden2 = self.lstm2.forward(hidden1)

            output = self.W_out.dot(hidden2).add(self.b_out)
            outputs.append(output)

        return outputs

    def predict_next_word(self, word_indices):
        outputs = self.forward(word_indices)
        last_output = outputs[-1]

        if self.temperature != 1.0:
            for i in range(last_output.rows):
                last_output.data[i][0] /= self.temperature

        probs = self._softmax(last_output)

        return self._sample_from_distribution(probs)

    def _softmax(self, logits):
        max_val = max(logits.to_array())

        exp_vals = Matrix(logits.rows, logits.cols)
        for i in range(logits.rows):
            for j in range(logits.cols):
                exp_vals.data[i][j] = math.exp(logits.data[i][j] - max_val)

        sum_exp = sum(exp_vals.to_array())

        probs = Matrix(logits.rows, logits.cols)
        for i in range(logits.rows):
            for j in range(logits.cols):
                probs.data[i][j] = exp_vals.data[i][j] / sum_exp

        return probs

    def _sample_from_distribution(self, probs):
        probs_array = probs.to_array()
        cumulative_probs = [0] * len(probs_array)
        cumulative_probs[0] = probs_array[0]

        for i in range(1, len(probs_array)):
            cumulative_probs[i] = cumulative_probs[i-1] + probs_array[i]

        r = random.random()
        for i, prob in enumerate(cumulative_probs):
            if r <= prob:
                return i

        return len(probs_array) - 1

    def generate_text(self, seed_text, max_length=50):
        word_indices = seed_text.copy()

        n_grams = {}
        n = 3

        for _ in range(max_length):
            context = word_indices[-self.context_size:] if len(word_indices) >= self.context_size else word_indices

            next_word_idx = self.predict_next_word(context)

            if len(word_indices) >= n:
                current_ngram = tuple(word_indices[-(n-1):] + [next_word_idx])
                if current_ngram in n_grams:
                    n_grams[current_ngram] += 1
                    if n_grams[current_ngram] > 2:
                        temp_backup = self.temperature
                        self.temperature = 1.2
                        next_word_idx = self.predict_next_word(context)
                        self.temperature = temp_backup
                else:
                    n_grams[current_ngram] = 1

            word_indices.append(next_word_idx)

        return word_indices[len(seed_text):]

    def train_step(self, input_indices, target_indices):
        outputs = self.forward(input_indices)

        for i, (output, target_idx) in enumerate(zip(outputs, target_indices)):
            target = Matrix(self.vocab_size, 1)
            target.data[target_idx][0] = 1

            error = target.subtract(self._softmax(output))

            gradients = error.multiply(self.learning_rate)

            self.W_out = self.W_out.add(gradients.dot(self.lstm2.h.transpose()))
            self.b_out = self.b_out.add(gradients)


class Tokenizer:
    def __init__(self):
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.vocab_size = 4

    def fit(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word_to_index:
                    self.word_to_index[word] = self.vocab_size
                    self.index_to_word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        return [self.word_to_index.get(word, self.word_to_index["<UNK>"])
                for word in text.split()]

    def decode(self, indices):
        return " ".join([self.index_to_word.get(idx, "<UNK>") for idx in indices])

    def batch_encode(self, texts, max_length=None):
        encoded = [self.encode(text) for text in texts]

        if max_length:
            encoded = [seq + [self.word_to_index["<PAD>"]] * (max_length - len(seq))
                       if len(seq) < max_length else seq[:max_length]
                       for seq in encoded]

        return encoded


class CharTokenizer:
    def __init__(self):
        self.char_to_index = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.index_to_char = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.vocab_size = 4

    def fit(self, texts):
        for text in texts:
            for char in text:
                if char not in self.char_to_index:
                    self.char_to_index[char] = self.vocab_size
                    self.index_to_char[self.vocab_size] = char
                    self.vocab_size += 1

    def encode(self, text):
        return [self.char_to_index.get(char, self.char_to_index["<UNK>"])
                for char in text]

    def decode(self, indices):
        return "".join([self.index_to_char.get(idx, "<UNK>") for idx in indices])


class SimpleLLM:
    def __init__(self, vocab_size=1000, embedding_dim=64, hidden_size=128):
        self.model = LanguageModel(vocab_size, embedding_dim, hidden_size)
        self.tokenizer = Tokenizer()
        self.context = []

        self.predefined_responses = {
            "hello": ["Hello! How can I help you today?",
                     "Hi there! I'm your AI assistant. What can I do for you?",
                     "Greetings! What would you like to talk about?"],
            "how are you": ["I'm functioning well, thank you for asking!",
                           "I'm doing great! How about you?",
                           "All systems operational. How can I assist you?"],
            "help": ["I can chat with you or generate simple ASCII images. Just ask!",
                    "I can have a conversation or create ASCII art images. What would you like?",
                    "I'm here to chat or generate images. What would you like to do?"],
            "thank": ["You're welcome!", "Happy to help!", "Anytime!"],
            "bye": ["Goodbye! Have a great day!", "See you later!", "Until next time!"]
        }

        self.enhanced_corpus = [
            "Hello, how are you today?",
            "I am doing well, thank you for asking.",
            "It's a pleasure to meet you.",
            "How may I assist you with your questions?",
            "I can help you with various tasks and answer questions.",
            "The weather is quite pleasant today.",
            "I enjoy having conversations with people.",
            "Learning new things is always interesting.",
            "Please let me know if you need any help.",
            "I can generate simple images based on your descriptions.",
            "Would you like to see an image of something specific?",
            "I was created to demonstrate neural networks.",
            "This system is built from scratch without external libraries.",
            "The implementation includes a basic language model.",
            "Neural networks can learn patterns from data.",
            "Language models predict the next word in a sequence.",
            "What kinds of topics are you interested in discussing?",
            "Image generation works by creating patterns from random noise.",
            "Tell me more about what you're working on.",
            "I'm designed to be helpful and informative.",
            "The field of artificial intelligence is advancing rapidly.",
            "Machine learning enables computers to learn from examples.",
            "Deep learning is a subset of machine learning.",
            "Would you like to know more about how I work?",
            "I'm continuously improving my responses.",
            "Feel free to ask me any questions you might have.",
            "My responses are generated based on patterns I've learned.",
            "What else would you like to talk about?",
            "I can discuss a variety of topics.",
            "Let me know if my response wasn't helpful."
        ]

    def train(self, training_data, epochs=5, batch_size=32):
        combined_training_data = training_data + self.enhanced_corpus

        print("Fitting tokenizer...")
        self.tokenizer.fit(combined_training_data)

        print(f"Vocabulary size: {self.tokenizer.vocab_size}")

        self.model = LanguageModel(self.tokenizer.vocab_size,
                                   self.model.embedding_dim,
                                   self.model.hidden_size)

        print("Encoding training data...")
        encoded_data = self.tokenizer.batch_encode(combined_training_data)

        print(f"Training on {len(encoded_data)} examples for {epochs} epochs...")
        for epoch in range(epochs):
            random.shuffle(encoded_data)

            for i in range(0, len(encoded_data), batch_size):
                batch = encoded_data[i:i + batch_size]

                for sequence in batch:
                    if len(sequence) < 2:
                        continue

                    inputs = sequence[:-1]
                    targets = sequence[1:]

                    self.model.train_step(inputs, targets)

                if i % (10 * batch_size) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, processed {i}/{len(encoded_data)} examples")

        print("Generating sample sentences to verify model:")
        for _ in range(3):
            seed = self.tokenizer.encode("I am")
            generated = self.model.generate_text(seed, max_length=10)
            print(f"Sample: I am {self.tokenizer.decode(generated)}")

    def respond(self, user_input, max_length=50):
        lowered_input = user_input.lower()

        for keyword, responses in self.predefined_responses.items():
            if keyword in lowered_input:
                return random.choice(responses)

        self.context.append("User: " + user_input)

        context_window = 3
        prompt = " ".join(self.context[-context_window*2:])
        prompt += " Assistant:"

        encoded_prompt = self.tokenizer.encode(prompt)

        generated_indices = self.model.generate_text(encoded_prompt, max_length=max_length*2)
        response = self.tokenizer.decode(generated_indices)

        end_markers = ['.', '!', '?', '\n']
        for marker in end_markers:
            if marker in response:
                response = response.split(marker)[0] + marker
                break

        if len(response.split()) < 3:
            variations = [
                "I understand. " + response,
                "I see. " + response,
                response + " Is there anything else you'd like to know?",
                response + " Can I help with anything else?"
            ]
            response = random.choice(variations)

        self.context.append("Assistant: " + response)

        return response
