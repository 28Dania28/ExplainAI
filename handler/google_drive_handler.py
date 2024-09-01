import requests

from handler.video_handler import VideoHandler


class GoogleDriveHandler(VideoHandler):
    def __init__(self, url):
        super().__init__(url)
        self.file_id = self.extract_drive_file_id(url)

    def extract_drive_file_id(self, url):
        if 'drive.google.com/file/d/' in url:
            return url.split('d/')[1].split('/')[0]
        return None

    def download_file(self, file_id, destination):
        file_url = f'https://drive.google.com/uc?export=download&id={file_id}'

        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    def get_transcript(self):
        if not self.file_id:
            raise ValueError("Invalid Google Drive file ID")

        file_name = f"downloaded_file"

        destination = f'/path/to/download/{file_name}'
        self.download_file(self.file_id, destination)

        return f"File downloaded to {destination}."
