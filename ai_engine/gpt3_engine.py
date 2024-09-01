import os

import openai

from ai_engine.ai_question_answerer import AIQuestionAnswerer


class GPT3Engine(AIQuestionAnswerer):
    GPT3_ENGINE = "text-davinci-003"
    OPENAI_API_KEY_ENV_VAR = 'OPENAI_API_KEY'

    def __init__(self, transcripts_dir):
        super().__init__(transcripts_dir)
        self.openai_api_key = os.getenv(self.OPENAI_API_KEY_ENV_VAR)
        if not self.openai_api_key:
            raise ValueError(f"Environment variable {self.OPENAI_API_KEY_ENV_VAR} is not set")
        openai.api_key = self.openai_api_key
        print("Using GPT-3 Model")

    def answer_question(self, question):
        if not self.transcript_text:
            return "No transcript text available for answering questions."
        response = openai.Completion.create(
            engine=self.GPT3_ENGINE,
            prompt=f"Context: {self.transcript_text}\n\nQuestion: {question}\nAnswer:",
            max_tokens=150
        )
        return response.choices[0].text.strip()
