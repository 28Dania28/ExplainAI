from ai_engine.hugging_face_engine import HuggingFaceEngine
from processor.content_processor import ContentProcessor


def process(video_links):
    processor = ContentProcessor(video_links)
    processor.process_videos()


def start_explain_ai_automated_questions():
    ai = HuggingFaceEngine(transcripts_dir='transcripts')
    print("ExplainAI Question Answerer is ready. Type 'exit' to quit.")

    ques_list = [
        "What is the main topic discussed in the transcript?",
        "What were the two main factors driving the market sentiment according to the transcript?",
        "Who is Katie Kaminsky, and what is her role?",
        "What specific job creation data from July is mentioned, and how did it impact the market?",
        "What interest rate change did Japan's Central Bank implement, and why is it significant?",
        "How did Nvidia’s share price change, and what implications does this have according to the discussion?",
        "What are the potential reasons for the market's negative reaction described in the transcript?",
        "How does the transcript suggest the Federal Reserve's actions could impact the market?",
        "What are the broader economic concerns raised in the transcript beyond the immediate market reaction?",
        "Can you provide a summary of the key points discussed about the current market conditions?",
        "How does the transcript suggest global economic factors are influencing the US market?",
        "What are the key takeaways regarding future economic expectations as mentioned in the transcript?",
        "How did the recent economic data influence market predictions according to the discussion?",
        "What is the significance of the September 18th date mentioned in the context of the Federal Reserve?",
        "What contrasting economic actions are highlighted between the US and other global markets?"
    ]

    for ques in ques_list:
        print("You : " + ques)
        answer = ai.answer_question(ques)
        print("ExplainAI:", answer)


def start_explain_ai():
    ai = HuggingFaceEngine(transcripts_dir='transcripts')
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
