from transformers import pipeline

from ai_engine.ai_question_answerer import AIQuestionAnswerer


class HuggingFaceEngine(AIQuestionAnswerer):
    MODELS = {
        '1': {'name': 'distilbert-base-uncased-distilled-squad', 'type': 'question-answering'},
        '2': {'name': 'bert-large-uncased-whole-word-masking-finetuned-squad', 'type': 'question-answering'},
        '3': {'name': 'facebook/bart-large', 'type': 'text-generation'},
        '4': {'name': 'deepset/roberta-base-squad2', 'type': 'question-answering'},
        '5': {'name': 'google/tapas-large-finetuned-wtq', 'type': 'question-answering'},
        '6': {'name': 'deepset/roberta-large-squad2', 'type': 'question-answering'},
        '7': {'name': 'allenai/longformer-large-4096', 'type': 'question-answering'},
        '8': {'name': 'bert-base-uncased', 'type': 'question-answering'},
        '9': {'name': 'bert-large-uncased', 'type': 'question-answering'},
        '10': {'name': 'bert-base-cased', 'type': 'question-answering'},
        '11': {'name': 'bert-large-cased', 'type': 'question-answering'},
        '12': {'name': 'bert-base-multilingual-cased', 'type': 'question-answering'},
        '13': {'name': 'bert-large-uncased-whole-word-masking', 'type': 'question-answering'},
        '14': {'name': 'bert-large-cased-whole-word-masking', 'type': 'question-answering'},
        '15': {'name': 't5-small', 'type': 'text2text-generation'},
        '16': {'name': 't5-base', 'type': 'text2text-generation'},
        '17': {'name': 't5-large', 'type': 'text2text-generation'},
        '18': {'name': 't5-3b', 'type': 'text2text-generation'},
        '19': {'name': 't5-11b', 'type': 'text2text-generation'},
        '20': {'name': 'facebook/bart-large-cnn', 'type': 'text-generation'},
        '21': {'name': 'facebook/bart-large-squad2', 'type': 'question-answering'},
        '22': {'name': 'albert-base-v2', 'type': 'question-answering'},
        '23': {'name': 'albert-large-v2', 'type': 'question-answering'},
        '24': {'name': 'albert-xlarge-v2', 'type': 'question-answering'},
        '25': {'name': 'albert-xxlarge-v2', 'type': 'question-answering'},
        '26': {'name': 'distilroberta-base', 'type': 'question-answering'},
        '27': {'name': 'distilbert-base-uncased', 'type': 'question-answering'},
        '28': {'name': 'google/electra-small-discriminator', 'type': 'question-answering'},
        '29': {'name': 'google/electra-base-discriminator', 'type': 'question-answering'},
        '30': {'name': 'google/electra-large-discriminator', 'type': 'question-answering'},
        '31': {'name': 'roberta-base', 'type': 'question-answering'},
        '32': {'name': 'roberta-large', 'type': 'question-answering'},
        '33': {'name': 'xlnet-base-cased', 'type': 'question-answering'},
        '34': {'name': 'xlnet-large-cased', 'type': 'question-answering'},
        '35': {'name': 'microsoft/mdeberta-v3-base', 'type': 'question-answering'},
        '36': {'name': 'microsoft/mdeberta-v3-large', 'type': 'question-answering'},
        '37': {'name': 'microsoft/mdeberta-v3-xlarge', 'type': 'question-answering'},
        '38': {'name': 'microsoft/mdeberta-v3-xxlarge', 'type': 'question-answering'},
        '39': {'name': 'nreimers/MiniLM-L6-H384-uncased', 'type': 'question-answering'},
        '40': {'name': 'microsoft/phi-2', 'type': 'text-generation'},
        '41': {'name': 'meta-llama/Meta-Llama-3-8B', 'type': 'text-generation'},
        '42': {'name': 'google/gemma-7b', 'type': 'text-generation'}
    }

    DEFAULT_MODEL_TYPE = '40'

    def __init__(self, transcripts_dir):
        super().__init__(transcripts_dir)
        model_info = self.MODELS[self.DEFAULT_MODEL_TYPE]
        self.model_name = model_info['name']
        self.model_type = model_info['type']

        self.qa_pipeline = pipeline(self.model_type, model=self.model_name, tokenizer=self.model_name, device=0)

        print("Using Hugging Face Model: " + self.model_name)

    def answer_question(self, question):
        if not self.transcript_text:
            return "No transcript text available for answering questions."

        if self.model_type == 'text2text-generation':
            input_text = f"question: {question} context: {self.transcript_text}"
            result = self.qa_pipeline(input_text, max_new_tokens=100)
            return result[0]['generated_text']
        elif self.model_type == 'text-generation':
            input_text = f"Question: {question} Context: {self.transcript_text}"
            result = self.qa_pipeline(input_text, max_new_tokens=100)
            return result[0]['generated_text']
        else:  # question-answering
            result = self.qa_pipeline(question=question, context=self.transcript_text)
            return result['answer']
