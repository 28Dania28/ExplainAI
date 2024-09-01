import json
import os

from transformers import pipeline


class AIQuestionAnswerer:
    MODELS = {
        '1': 'distilbert-base-uncased-distilled-squad',
        '2': 'bert-large-uncased-whole-word-masking-finetuned-squad',
        '3': 'facebook/bart-large',
        '4': 'deepset/roberta-base-squad2',
        '5': 'google/tapas-large-finetuned-wtq'
    }

    def __init__(self, transcripts_dir):
        self.transcripts_dir = transcripts_dir
        self.transcript_text = self.load_transcripts()
        self.model_name = self.MODELS.get(3, 'facebook/bart-large')
        self.qa_pipeline = self.get_model_pipeline()

    def get_model_pipeline(self):
        return pipeline('question-answering', model=self.model_name, device=0)

    def load_transcripts(self):
        combined_text = ""
        for file_name in os.listdir(self.transcripts_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.transcripts_dir, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    combined_text += data.get('transcript', '') + " "
        return combined_text

    def answer_question(self, question):
        if not self.transcript_text:
            return "No transcript text available for answering questions."

        result = self.qa_pipeline(question=question, context=self.transcript_text)
        return result['answer']
