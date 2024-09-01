import json

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.formatters import JSONFormatter

from handler.video_handler import VideoHandler


class YouTubeHandler(VideoHandler):
    def __init__(self, url):
        super().__init__(url)
        self.video_id = self.extract_youtube_video_id(url)

    def extract_youtube_video_id(self, url):
        if 'youtube.com/watch?v=' in url:
            return url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1]
        return None

    def get_transcript(self):
        if not self.video_id:
            raise ValueError("Invalid YouTube video ID")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            formatter = JSONFormatter()
            transcript_json_string = formatter.format_transcript(transcript_list)
            transcript_json = json.loads(transcript_json_string)
            return transcript_json
        except TranscriptsDisabled:
            return "Transcript is disabled for this video"
        except Exception as e:
            return f"An error occurred: {str(e)}"
