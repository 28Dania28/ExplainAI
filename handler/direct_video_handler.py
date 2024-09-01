import os

import whisper
import yt_dlp

from handler.video_handler import VideoHandler


class DirectVideoHandler(VideoHandler):
    def __init__(self, url):
        super().__init__(url)
        self.model = whisper.load_model("base")

    def download_audio(self, output_filename='audio.wav'):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'ffmpeg_location': '/opt/homebrew/bin',
            'verbose': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            print("Audio downloaded and extracted successfully.")
        except Exception as e:
            print(f"An error occurred while downloading audio: {e}")

    def transcribe_audio(self, audio_filename='audio.wav'):
        result = self.model.transcribe(audio_filename)
        return result['text']

    def get_transcript(self):
        audio_filename = 'audio.wav'

        self.download_audio(audio_filename)
        transcript = self.transcribe_audio(audio_filename)

        os.remove(audio_filename)
        return transcript
