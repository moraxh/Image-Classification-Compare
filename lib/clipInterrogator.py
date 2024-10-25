import torch
from PIL import Image
from clip_interrogator import Config, Interrogator

config = Config(clip_model_name="ViT-L-14/openai", download_cache=True)
config.device = "cuda" if torch.cuda.is_available() else "cpu"
# config.apply_low_vram_defaults()
ci = Interrogator(config)

def predict(img_name, mode):
  image = Image.open(img_name).convert('RGB')
  if mode == 'best':
      return ci.interrogate(image)
  elif mode == 'classic':
      return ci.interrogate_classic(image)
  elif mode == 'fast':
      return ci.interrogate_fast(image)
  elif mode == 'negative':
      return ci.interrogate_negative(image)