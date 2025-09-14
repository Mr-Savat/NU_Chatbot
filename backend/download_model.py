import os
import requests
import zipfile

def download_model():
    model_url = "https://github.com/Mr-Savat/NU_Chatbot/releases/download/v1.0/gpt2-norton.zip"
    model_path = "gpt2-norton"
    
    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(model_url)
        with open("gpt2-norton.zip", "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile("gpt2-norton.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        
        os.remove("gpt2-norton.zip")
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")

if __name__ == "__main__":
    download_model()