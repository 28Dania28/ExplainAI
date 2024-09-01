import json
import os
import urllib.parse

from handler.direct_video_handler import DirectVideoHandler
from handler.google_drive_handler import GoogleDriveHandler
from handler.youtube_handler import YouTubeHandler


class ContentProcessor:
    def __init__(self, urls, transcripts_dir='transcripts'):
        self.urls = urls
        self.transcripts_dir = transcripts_dir
        self.handlers = {
            'youtube': YouTubeHandler,
            'google_drive': GoogleDriveHandler,
            'direct': DirectVideoHandler
        }
        self.transcripts = []

    def get_video_type(self, url):
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'drive.google.com' in url:
            return 'google_drive'
        elif url.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return 'direct'
        else:
            return 'unknown'

    def process_videos(self):
        if not os.path.exists(self.transcripts_dir):
            os.makedirs(self.transcripts_dir)

        for url in self.urls:
            video_type = self.get_video_type(url)
            handler_class = self.handlers.get(video_type, None)
            if handler_class:
                handler = handler_class(url)
                transcript = handler.get_transcript()
                self.transcripts.append({'url': url, 'transcript': transcript})
                self.save_transcript(url, transcript)
            else:
                self.transcripts.append({'url': url, 'transcript': 'Unsupported video type'})
                self.save_transcript(url, 'Unsupported video type')

    def save_transcript(self, url, transcript):
        file_name = self.generate_file_name(url)
        file_path = os.path.join(self.transcripts_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump({'url': url, 'transcript': transcript}, file)

    def generate_file_name(self, url):
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path.replace('/', '_').replace('.', '_')
        query = urllib.parse.parse_qs(parsed_url.query)
        video_id = query.get('v', ['unknown'])[0]
        return f"{path}_{video_id}.json"

    def get_transcripts(self):
        return self.transcripts
