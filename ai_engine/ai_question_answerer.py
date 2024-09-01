import json
import os
import re


class AIQuestionAnswerer:
    def __init__(self, transcripts_dir):
        self.transcripts_dir = transcripts_dir
        self.transcript_text = self.load_transcripts()

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
        raise NotImplementedError("Subclasses should implement this!")
