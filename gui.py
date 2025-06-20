import tkinter as tk
from tkinter import scrolledtext, Frame, Button, Label, Entry
import threading
import time

from language_model import SimpleLLM
from image_generator import ImageGenerator, TextToImageGenerator
from neural_network import Matrix

class AIAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        print("Initializing AI Assistant...")
        self.init_models()

        self.setup_gui()

        self.models_ready = False

        self.train_thread = threading.Thread(target=self.train_models)
        self.train_thread.daemon = True
        self.train_thread.start()

    def init_models(self):
        vocab_size = 800
        embedding_dim = 64
        hidden_size = 128
        latent_dim = 50
        image_width = 20
        image_height = 20

        self.llm = SimpleLLM(vocab_size, embedding_dim, hidden_size)

        self.image_generator = ImageGenerator(latent_dim, image_width, image_height)

        self.text_to_image = TextToImageGenerator(self.llm, self.image_generator)

        self.example_phrases = [
            "Hello, how are you?",
            "I'm doing well, thank you for asking.",
            "What can I help you with today?",
            "I can chat with you and generate simple images.",
            "What kind of image would you like me to create?",
            "I can try to generate a simple picture based on your description.",
            "The weather is nice today.",
            "I'm a simple AI assistant built from scratch.",
            "This system has no external dependencies.",
            "I was created as a demonstration of neural networks.",
            "I can understand basic text and generate responses.",
            "My image generation is very basic but demonstrates the concept.",
            "What would you like to talk about?",
            "I'm learning to communicate better.",
            "Tell me more about what you're interested in.",
            "I'll do my best to assist you with your questions."
        ]

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
            "a simple landscape"
        ]

        self.example_phrases.extend(self.llm.enhanced_corpus)

    def setup_gui(self):
        top_frame = Frame(self.root, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=10)

        middle_frame = Frame(self.root)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        bottom_frame = Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)

        self.title_label = Label(top_frame, text="AI Assistant", font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.title_label.pack(side=tk.LEFT, padx=5)

        self.status_label = Label(top_frame, text="Status: Initializing...",
                                 font=("Arial", 10), fg="blue", bg="#f0f0f0")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        chat_image_frame = Frame(middle_frame)
        chat_image_frame.pack(fill=tk.BOTH, expand=True)

        chat_frame = Frame(chat_image_frame)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        chat_label = Label(chat_frame, text="Conversation", font=("Arial", 12))
        chat_label.pack(anchor="w", padx=5, pady=5)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD,
                                                     font=("Consolas", 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        image_frame = Frame(chat_image_frame, width=300)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        image_frame.pack_propagate(False)

        image_label = Label(image_frame, text="Generated Image", font=("Arial", 12))
        image_label.pack(anchor="w", padx=5, pady=5)

        self.image_display = scrolledtext.ScrolledText(image_frame, wrap=tk.WORD,
                                                      font=("Courier New", 10), width=30)
        self.image_display.pack(fill=tk.BOTH, expand=True)
        self.image_display.config(state=tk.DISABLED)

        Label(bottom_frame, text="Your message:", bg="#f0f0f0").pack(anchor="w", padx=5, pady=5)

        input_button_frame = Frame(bottom_frame, bg="#f0f0f0")
        input_button_frame.pack(fill=tk.X, expand=True)

        self.user_input = Entry(input_button_frame, font=("Arial", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.config(state=tk.DISABLED)

        self.send_button = Button(input_button_frame, text="Send", command=self.send_message,
                                 font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", width=8)
        self.send_button.pack(side=tk.LEFT, padx=(0, 5))
        self.send_button.config(state=tk.DISABLED)

        self.image_button = Button(input_button_frame, text="Generate Image",
                                  command=self.generate_image,
                                  font=("Arial", 10), bg="#2196F3", fg="white")
        self.image_button.pack(side=tk.LEFT)
        self.image_button.config(state=tk.DISABLED)

        self.update_chat_display("System", "Welcome to the AI Assistant! The models are initializing and training...\n" +
                                "This may take a minute or two. Please wait until the status shows 'Ready'.")

    def train_models(self):
        self.update_status("Training language model...")

        try:
            self.llm.train(self.example_phrases, epochs=3)

            self.models_ready = True
            self.update_status("Ready")

            self.root.after(0, self.enable_ui)

            self.root.after(0, lambda: self.update_chat_display(
                "AI", "Hello! I'm your AI assistant built from scratch with no external modules.\n" +
                "I can chat with you and generate simple ASCII images. How can I help you today?"))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Training error: {e}")

    def enable_ui(self):
        self.user_input.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.image_button.config(state=tk.NORMAL)

    def update_status(self, message):
        def _update():
            self.status_label.config(text=f"Status: {message}")

        self.root.after(0, _update)

    def update_chat_display(self, sender, message):
        def _update():
            self.chat_display.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")

            if sender == "AI":
                self.chat_display.insert(tk.END, f"{timestamp} AI: ", "ai_tag")
                self.chat_display.tag_configure("ai_tag", foreground="green", font=("Arial", 10, "bold"))
            elif sender == "You":
                self.chat_display.insert(tk.END, f"{timestamp} You: ", "user_tag")
                self.chat_display.tag_configure("user_tag", foreground="blue", font=("Arial", 10, "bold"))
            else:
                self.chat_display.insert(tk.END, f"{timestamp} {sender}: ", "system_tag")
                self.chat_display.tag_configure("system_tag", foreground="gray", font=("Arial", 10, "italic"))

            self.chat_display.insert(tk.END, f"{message}\n\n")
            self.chat_display.see(tk.END)
            self.chat_display.config(state=tk.DISABLED)

        self.root.after(0, _update)

    def update_image_display(self, ascii_art):
        def _update():
            self.image_display.config(state=tk.NORMAL)
            self.image_display.delete(1.0, tk.END)
            self.image_display.insert(tk.END, ascii_art)
            self.image_display.config(state=tk.DISABLED)

        self.root.after(0, _update)

    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if not message:
            return

        self.user_input.delete(0, tk.END)

        self.update_chat_display("You", message)

        if ("generate" in message.lower() and
            ("image" in message.lower() or "picture" in message.lower())):
            self.process_image_request(message)
        else:
            threading.Thread(target=self.process_message, args=(message,)).start()

    def process_message(self, message):
        try:
            self.update_status("Thinking...")

            response = self.llm.respond(message)

            if not response or len(response.strip()) < 3:
                response = "I'm still learning to communicate. Can you ask me something else?"

            self.update_chat_display("AI", response)

            self.update_status("Ready")

        except Exception as e:
            self.update_chat_display("System", f"Error processing message: {str(e)}")
            self.update_status("Error")
            print(f"Error processing message: {e}")

    def generate_image(self, event=None):
        description = self.user_input.get().strip()

        if description:
            self.update_chat_display("You", f"Generate an image of: {description}")

            self.user_input.delete(0, tk.END)

            threading.Thread(target=self.process_image_request,
                         args=(f"Generate an image of {description}",)).start()
        else:
            self.update_chat_display("System", "What would you like me to generate an image of?")

            self.send_button.config(text="Generate", command=self.generate_from_description)

            self._original_send_function = self.send_message
            self.send_message = self.generate_from_description

            self.user_input.focus_set()

    def generate_from_description(self, event=None):
        description = self.user_input.get().strip()
        if not description:
            return

        self.user_input.delete(0, tk.END)

        self.update_chat_display("You", f"Generate an image of: {description}")

        self.send_button.config(text="Send", command=self._original_send_function)
        self.send_message = self._original_send_function

        threading.Thread(target=self.process_image_request,
                         args=(f"Generate an image of {description}",)).start()

    def process_image_request(self, message):
        try:
            self.update_status("Generating image...")

            description = ""
            if "of " in message.lower():
                description = message.lower().split("of ")[1].strip()
            elif "showing " in message.lower():
                description = message.lower().split("showing ")[1].strip()

            if not description:
                self.update_chat_display("AI", "What would you like me to generate an image of?")
                self.update_status("Ready")
                return

            image = self.text_to_image.generate_from_text(description)

            ascii_art = self.image_generator.render_ascii(image)

            self.update_chat_display("AI", f"Here's a simple image of '{description}':")
            self.update_image_display(ascii_art)

            self.update_status("Ready")

        except Exception as e:
            self.update_chat_display("System", f"Error generating image: {str(e)}")
            self.update_status("Error")
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = AIAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
