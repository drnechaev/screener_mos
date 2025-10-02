import os
import sys
import argparse
import requests

class FileUploader:
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.token = token
        self.session = requests.Session()
    
    def find_zip_files(self, directory: str):
        """Находит все ZIP файлы в директории"""
        zip_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.zip'):
                    zip_files.append(os.path.join(root, file))
        return zip_files
    
    def upload_file(self, file_path: str):
        """Загружает файл на сервер"""
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file, 'application/zip')}
                data = {'token': self.token}
                response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def upload_all_files(self, directory: str):
        """Загружает все ZIP файлы из директории"""
        zip_files = self.find_zip_files(directory)
        
        if not zip_files:
            print("No ZIP files found")
            return
        
        print(f"Found {len(zip_files)} files")
        
        success_count = 0
        for file_path in zip_files:
            filename = os.path.basename(file_path)
            print(f"Uploading: {filename}")
            
            result = self.upload_file(file_path)
            if result['success']:
                print(f"✓ Success: {filename}")
                success_count += 1
            else:
                print(f"✗ Error: {filename}")
            
            print("-" * 30)
        
        print(f"Completed: {success_count}/{len(zip_files)} successful")

def main():
    parser = argparse.ArgumentParser(description='Upload ZIP files to server')
    parser.add_argument('directory', help='Directory with ZIP files')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found")
        sys.exit(1)
    
    uploader = FileUploader("http://screener.airi.net:7534/v1/upload", "MOSCOWTOCKEN2025")
    uploader.upload_all_files(args.directory)

if __name__ == "__main__":
    main()