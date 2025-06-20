import time
import random

from neural_network import Matrix
from language_model import SimpleLLM
from image_generator import ImageGenerator, TextToImageGenerator

class AIAssistant:
    """An AI assistant that can converse and generate images"""

    def __init__(self, vocab_size=1000, embedding_dim=64, hidden_size=128,
                 latent_dim=100, image_width=28, image_height=28):
        print("Initializing AI Assistant...")
        print("This is a simplified AI system built from scratch with no external modules.")

        # Initialize language model
        self.llm = SimpleLLM(vocab_size, embedding_dim, hidden_size)

        # Initialize image generator
        self.image_generator = ImageGenerator(latent_dim, image_width, image_height)

        # Initialize text-to-image generator
        self.text_to_image = TextToImageGenerator(self.llm, self.image_generator)

        # Training data for the language model - expanded with more complete sentences
        self.example_phrases = [
            # Basic conversation starters
            "Hello, how are you?",
            "I'm doing well, thank you for asking.",
            "What can I help you with today?",
            "I can chat with you and generate simple images.",
            "What kind of image would you like me to create?",
            "I can try to generate a simple picture based on your description.",

            # General knowledge and capabilities
            "The weather is nice today.",
            "I'm a simple AI assistant built from scratch.",
            "This system has no external dependencies.",
            "I was created as a demonstration of neural networks.",
            "I can understand basic text and generate responses.",
            "My image generation is very basic but demonstrates the concept.",

            # Conversation continuers
            "What would you like to talk about?",
            "I'm learning to communicate better.",
            "Tell me more about what you're interested in.",
            "I'll do my best to assist you with your questions.",

            # Added more complete sentences with proper grammar
            "Artificial intelligence is a fascinating field of study.",
            "Neural networks are inspired by the human brain's structure.",
            "Machine learning allows computers to learn from examples.",
            "The field of computer science continues to evolve rapidly.",
            "Language models predict the next word in a sequence.",
            "Text generation works by predicting likely word sequences.",
            "Please let me know if my responses are helpful to you.",
            "I appreciate your patience as I'm still learning.",
            "Would you like to know more about how I work?",
            "Each response is generated from patterns I've learned.",
            "The quality of my responses improves with more training data.",
            "My training involves learning word relationships and patterns.",
            "Thank you for conversing with me today.",
            "Image generation converts text descriptions into visual patterns.",
            "I hope my responses have been clear and useful.",
            "Feel free to ask me anything you'd like to know."
        ]

        # Example image prompts and descriptions
        self.image_examples = [
            "a circle",
            "a square",
            "a simple face",
            "a tree",
            "a house",
            "a cat",
            "a sun",
            "a mountain",
            "a cloud",
            "a simple landscape",
            # Add more specific descriptions
            "a smiling face with two eyes",
            "a house with a door and two windows",
            "a tree with branches and leaves",
            "a cat with whiskers and pointed ears",
            "a mountain range with a sun above it",
            "a boat floating on water",
            "a simple flower with petals",
            "a butterfly with two wings",
            "a bird flying in the sky",
            "a car with four wheels"
        ]

        # Add template responses for common queries
        self.fallback_responses = [
            "I'm still learning to communicate. Can you ask me something else?",
            "That's interesting. Tell me more.",
            "I understand. Is there anything else you'd like to discuss?",
            "I'm a simple AI trying my best to respond appropriately.",
            "I'm not sure I understood that correctly. Could you rephrase?",
            "I'm learning to provide better responses. What else would you like to know?",
            "That's a good question. Let me try to answer that for you.",
            "I'm processing what you said. Could you provide more details?",
            "I appreciate your patience as I continue to learn and improve.",
            "I'm designed to be helpful. How else can I assist you today?"
        ]

    def train(self, epochs=3):
        """Train the language and image models"""
        print("Training language model...")
        # Increase epochs for better language quality
        self.llm.train(self.example_phrases, epochs=epochs)

        print("Training complete!")

    def respond_to_text(self, user_input):
        """Generate a response to user text input"""
        if not user_input.strip():
            return "I didn't receive any input. How can I help you?"

        # Check for image generation request
        if "generate" in user_input.lower() and ("image" in user_input.lower() or "picture" in user_input.lower()):
            # Extract description after "of" or "showing"
            description = ""
            if "of " in user_input.lower():
                description = user_input.lower().split("of ")[1].strip()
            elif "showing " in user_input.lower():
                description = user_input.lower().split("showing ")[1].strip()

            if description:
                print(f"Generating image of: {description}")
                return self.generate_image(description)
            else:
                return "What would you like me to generate an image of?"

        # Regular conversation
        response = self.llm.respond(user_input)

        # If model isn't well trained, it might return empty or nonsensical responses
        # In that case, fall back to template responses
        if not response or len(response.strip()) < 3:
            response = random.choice(self.fallback_responses)

        return response

    def generate_image(self, description):
        """Generate an image based on text description"""
        print(f"Generating image from description: '{description}'")

        # Get an image from the generator
        image = self.text_to_image.generate_from_text(description)

        # Render as ASCII for display in terminal
        ascii_art = self.image_generator.render_ascii(image)

        return f"Here's a simple image based on your description:\n\n{ascii_art}"


def main():
    """Main function to run the AI assistant"""
    print("Starting AI Assistant...")
    print("Initializing models (this may take a moment)...")

    # Create assistant with slightly larger model for better language capability
    assistant = AIAssistant(vocab_size=800, embedding_dim=64, hidden_size=128,
                           latent_dim=50, image_width=20, image_height=20)

    # Train the assistant with more epochs
    print("Training assistant on example data...")
    assistant.train(epochs=4)

    print("\n" + "="*50)
    print("AI Assistant is ready!")
    print("You can chat with it and ask it to generate simple images.")
    print("Example: 'Generate an image of a house'")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    # Main conversation loop
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI: Goodbye! Have a great day!")
            break

        # Get response
        start_time = time.time()
        response = assistant.respond_to_text(user_input)
        end_time = time.time()

        # Display response with processing time
        print(f"AI: {response}")
        print(f"(Response generated in {end_time - start_time:.2f} seconds)")
        print()


if __name__ == "__main__":
    main()


