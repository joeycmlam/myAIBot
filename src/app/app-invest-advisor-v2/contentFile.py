import requests

# Set your API key
api_key = 'your-api-key'

# Define the file path and other parameters
file_path = 'path/to/your/file.txt'
purpose = 'fine-tune'  # or 'answers', 'classifications', etc.

# Open the file and make the request
with open(file_path, 'rb') as file:
    response = requests.post(
        'https://api.openai.com/v1/files',
        headers={
            'Authorization': f'Bearer {api_key}',
        },
        files={
            'file': file,
        },
        data={
            'purpose': purpose,
        }
    )

# Check the response
if response.status_code == 200:
    print('File uploaded successfully:', response.json())
else:
    print('Failed to upload file:', response.text)