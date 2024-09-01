from ai_question_answerer import AIQuestionAnswerer
from processor.content_processor import ContentProcessor


def process(video_links):
    processor = ContentProcessor(video_links)
    processor.process_videos()


def start_explain_ai():
    ai = AIQuestionAnswerer(transcripts_dir='transcripts')

    print("ExplainAI Question Answerer is ready. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break

        answer = ai.answer_question(question)
        print("ExplainAI:", answer)


def main():
    video_links = [
        "https://www.youtube.com/watch?v=ijEJnEghYKY"
    ]

    process(video_links)
    start_explain_ai()


if __name__ == "__main__":
    main()
