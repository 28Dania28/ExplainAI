import json
import os
import re

from transformers import pipeline


class AIQuestionAnswerer:
    MODELS = {
        '1': 'distilbert-base-uncased-distilled-squad',
        '2': 'bert-large-uncased-whole-word-masking-finetuned-squad',
        '3': 'facebook/bart-large',
        '4': 'deepset/roberta-base-squad2',
        '5': 'google/tapas-large-finetuned-wtq',
        '6': 'deepset/roberta-large-squad2',
        '7': 'allenai/longformer-large-4096'
    }

    def __init__(self, transcripts_dir):
        self.transcripts_dir = transcripts_dir
        self.transcript_text = self.load_transcripts()

        self.model_name = self.MODELS.get('4', 'deepset/roberta-base-squad2')
        self.qa_pipeline = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name,
                                    device=0)

    def load_transcripts(self):
        combined_text = ""
        for file_name in os.listdir(self.transcripts_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.transcripts_dir, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    combined_text += self.preprocess_transcript(data) + " "
        print("Transcript Context : " + combined_text)
        return combined_text

    def preprocess_transcript(self, data):
        segments = data.get('transcript', [])
        full_text = ' '.join(segment['text'] for segment in segments)
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        return cleaned_text

    def answer_question(self, question):
        if not self.transcript_text:
            return "No transcript text available for answering questions."

        result = self.qa_pipeline(question=question, context=self.transcript_text)
        return result['answer']
