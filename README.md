# Neural Network from Scratch

A custom neural network implementation built entirely from the ground up, using no external modules or dependencies. This project includes a language model and image generation system with a graphical user interface.

## Overview

This project demonstrates how to build core neural network components, language models, and image generation capabilities using only Python's standard library. It's designed as an educational tool to understand the fundamental concepts behind neural networks without relying on frameworks like TensorFlow or PyTorch.

## Features

- **Neural Network Framework**: Custom matrix operations, activation functions, feed-forward networks, LSTMs, and embeddings
- **Language Model**: A simple text generation model with LSTM layers for conversation
- **Image Generator**: Generates ASCII art images from text descriptions
- **Graphical User Interface**: Interact with the AI through a simple Tkinter-based interface
- **Command-line Interface**: Alternative way to interact with the system

## Project Structure

- `neural_network.py`: Core neural network components and matrix operations
- `language_model.py`: Implementation of the language model and tokenization
- `image_generator.py`: Image generation system and text-to-image conversion
- `main.py`: Command-line interface for the AI assistant
- `gui.py`: Graphical user interface for easier interaction

## How to Use

### Requirements

- Python 3.6+ (tested on Python 3.13)
- No external packages required!

### Running the GUI

```bash
python gui.py
```

The GUI interface allows you to:
1. Chat with the AI assistant
2. Generate images from text descriptions
3. See the conversation history and generated images

### Running the Command-line Interface

```bash
python main.py
```

The command-line interface supports:
1. Text conversations with the AI
2. Image generation through text commands like "Generate an image of a house"
3. Exit the program by typing "exit", "quit", or "bye"

## How It Works

### Neural Network

The neural network implementation includes:
- Matrix class for mathematical operations
- Activation functions (sigmoid, tanh, ReLU, leaky ReLU)
- Feedforward neural networks
- LSTM cells for sequence modeling
- Embedding layers for word representations

### Language Model

The language model consists of:
- Tokenizer to convert text to numerical indices
- Embedding layer to represent words as vectors
- LSTM layers to learn sequential patterns
- Softmax output to predict next tokens
- Temperature-based sampling for text generation

### Image Generator

The image generation system uses:
- A multilayer perceptron to transform latent vectors into images
- ASCII art rendering to display generated images
- Text-to-image mapping via the language model

## Performance Notes

Since this system is built from scratch without optimized libraries:
- Training and inference are slower than with specialized frameworks
- The model sizes are kept small to maintain reasonable performance
- Image quality is limited and represented through ASCII art
- Language capabilities are basic but demonstrate the core concepts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Derek Yuan
- Grigory Grishin

## Acknowledgements

This project is built for educational purposes to demonstrate how neural networks, language models, and image generation systems work at a fundamental level.
