import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
__dir__ = f"{__dir__}"

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(f"{__dir__}/{model_file}", os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url + f' -P "{__dir__}"')

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(f"{__dir__}/{model_file}", map_location=lambda storage, loc: storage, weights_only=True)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
    trn.Resize((256,256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(f"{__dir__}/{file_name}", os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url + f' -P "{__dir__}"')
classes = list()
with open(f"{__dir__}/{file_name}") as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

def predict(img_name, n=5):
  img = Image.open(img_name)
  input_img = V(centre_crop(img).unsqueeze(0))

  # forward pass
  logit = model.forward(input_img)
  h_x = F.softmax(logit, 1).data.squeeze()
  probs, idx = h_x.sort(0, True)

  prediction = {}

  # output the prediction

  for i in range(0, n):
    prediction[classes[idx[i]]] = round(probs[i].item(), 4)

  return prediction