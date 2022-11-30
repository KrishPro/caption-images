from flask import Flask, jsonify, request
import torchvision.transforms as T
from tokenizers import Tokenizer
from io import BytesIO
from demo import ViT
import PIL.Image
import base64
import torch

def load_image(encoded_img: str):
    image: PIL.Image.Image = PIL.Image.open(BytesIO(base64.b64decode(encoded_img))).resize((256, 256))

    image: torch.Tensor = T.ToTensor()(image)

    return image

tokenizer: Tokenizer = Tokenizer.from_file('vocab/vocab.json')

model = ViT.from_ckpt("Output/model.ckpt").eval()

app = Flask(__name__)

@app.route('/')
def home_page():
    return "API is running !"

@app.route('/caption', methods=['POST'])
def caption():

    image = load_image(request.data)

    caption: torch.Tensor = model.generate_caption(image)

    caption = tokenizer.decode(caption.tolist())

    response = jsonify({'caption': caption})
    
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

app.run(host='0.0.0.0')