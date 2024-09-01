from transformers import pipeline

from ai_engine.ai_question_answerer import AIQuestionAnswerer


class HuggingFaceEngine(AIQuestionAnswerer):
    MODELS = {
        '1': 'distilbert-base-uncased-distilled-squad',
        '2': 'bert-large-uncased-whole-word-masking-finetuned-squad',
        '3': 'facebook/bart-large',
        '4': 'deepset/roberta-base-squad2',
        '5': 'google/tapas-large-finetuned-wtq',
        '6': 'deepset/roberta-large-squad2',
        '7': 'allenai/longformer-large-4096'
    }

    DEFAULT_MODEL_TYPE = '4'

    def __init__(self, transcripts_dir):
        super().__init__(transcripts_dir)
        self.model_name = self.MODELS[self.DEFAULT_MODEL_TYPE]
        self.qa_pipeline = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name, device=0)
        print("Using Hugging Face Model: " + self.model_name)

    def answer_question(self, question):
        if not self.transcript_text:
            return "No transcript text available for answering questions."
        result = self.qa_pipeline(question=question, context=self.transcript_text)
        return result['answer']
