from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    BASE_MODEL_NAME,
    MODEL_OUTPUT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)

from src.safety import (
    SafetyChecker,
    clean_model_response,
    add_medical_disclaimer,
)


SYSTEM_INSTRUCTION = (
    "You are an Alzheimer's support chatbot. "
    "Give helpful, simple, compassionate, and safe information for patients and caregivers. "
    "Do not claim to diagnose disease. "
    "Do not replace doctors. "
    "For emergencies or medication decisions, advise contacting a qualified healthcare professional."
)


class AlzheimerChatbot:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_OUTPUT_DIR
        self.safety_checker = SafetyChecker()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            model_path = str(self.model_dir)
            print(f"Loading fine-tuned model from: {model_path}")
        else:
            model_path = BASE_MODEL_NAME
            print(
                f"Fine-tuned model not found at {self.model_dir}. "
                f"Loading base model instead: {BASE_MODEL_NAME}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, user_input: str) -> str:
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"### User:\n{user_input}\n\n"
            f"### Assistant:\n"
        )

    def generate_raw_response(self, user_input: str) -> str:
        prompt = self.build_prompt(user_input)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )

        if "### Assistant:" in generated_text:
            response = generated_text.split("### Assistant:")[-1]
        else:
            response = generated_text.replace(prompt, "")

        if "### User:" in response:
            response = response.split("### User:")[0]

        if "<|endoftext|>" in response:
            response = response.split("<|endoftext|>")[0]

        return response.strip()

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()

        if not user_input:
            return "Please type a question."

        safety_result = self.safety_checker.check(user_input)

        if not safety_result.allowed:
            return safety_result.message

        raw_answer = self.generate_raw_response(user_input)
        cleaned_answer = clean_model_response(raw_answer)

        if not cleaned_answer or len(cleaned_answer) < 10:
            cleaned_answer = (
                "I understand your question. Alzheimer's symptoms and caregiving concerns "
                "can be difficult. Could you provide a little more detail so I can give a more helpful general answer?"
            )

        if safety_result.message:
            cleaned_answer = f"{safety_result.message}\n\n{cleaned_answer}"

        final_answer = add_medical_disclaimer(cleaned_answer)

        return final_answer


def interactive_chat():
    chatbot = AlzheimerChatbot()

    print("\nAlzheimer's Chatbot")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Bot: Goodbye.")
            break

        answer = chatbot.chat(user_input)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    interactive_chat()