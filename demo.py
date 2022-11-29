"""
Written by KrishPro @ KP

filename: `eval.py`
"""


from model import ViT
import torchvision.transforms as T
from tokenizers import Tokenizer
import torch.nn.functional as F
import PIL.Image
import torch

@ViT.add_method
@torch.no_grad()
def generate_caption(self: ViT, image: torch.Tensor, max_len=256):
    sos_token, eos_token = 1, 2

    image = image.unsqueeze(0)
    # image.shape: (1, 3, 256, 256)

    image: torch.Tensor = self.input(image)
    # image.shape: (1, 1, 256, 256)

    image = image.squeeze(0).reshape(1, 16, 16, 16, 16).permute(0, 1, 3, 2, 4).reshape(1, 256, 256)
    # image.shape: (1, 256, 256)

    image = self.pos_embedding(image)
    # image.shape: (1, 256, 256)

    for layer in self.encoder_layers:
        image = layer(image)
        # image.shape: (1, 256, 256)

    caption = torch.tensor([sos_token]).reshape(1, -1)
    prediction = torch.tensor([-1])

    i = 0
    while (prediction.item() != eos_token) and (i < (max_len-1)):
        i += 1

        # caption.shape: (1, T)
        embeddings = self.pos_embedding(self.tgt_embedding(caption))
        # embeddings.shape: (1, T, 256)

        for layer in self.decoder_layers:
            embeddings = layer(embeddings, image)
            # embeddings.shape: (1, T, 256)

        prediction  = F.softmax(self.output(embeddings[:, -1, :]), dim=1).argmax(1)
        # prediction.shape: (1,)

        caption = torch.cat([caption, prediction.unsqueeze(0)], dim=1)
        # caption.shape: (1, T+1)

    return caption.squeeze(0)    
    

def load_image(img_path: str):
    image = PIL.Image.open(img_path).resize((256, 256))

    image = T.ToTensor()(image)

    return image

def main():
    tokenizer: Tokenizer = Tokenizer.from_file('vocab/vocab.json')

    image = load_image('/home/krish/Datasets/flickr30k/images/3030079705.jpg') 

    model = ViT.from_ckpt('Output/model.ckpt').eval()

    caption: torch.Tensor = model.generate_caption(image)

    caption = tokenizer.decode(caption.tolist())

    print("Caption: ", caption)


if __name__ == '__main__':
    main()
