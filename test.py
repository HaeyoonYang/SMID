import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import argparse
# import torchsummary
from torchvision.utils import save_image
from EDSR import *
from preprocessing import *
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
np.random.seed(0)

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--noise_sigma', default=25, type=int, help='noise level')
args = parser.parse_args()
save_dir = os.path.join('results', str(args.noise_sigma))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model = EDSR()
# torchsummary.summary(model, (12, 100, 100))
state_dict = torch.load('models/model_007.pth')
model.load_state_dict(state_dict)
model.cuda()
model.eval()

test_data = TestData(data_path='data', sigma=args.noise_sigma)
# test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

max_PSNR = 0
max_name = ''
min_PSNR = 40
min_name = ''
total_PSNR = 0
for _, data in enumerate(test_data):
    clean_img, noisy_img, name = data
    noisy_img = torch.unsqueeze(noisy_img, 0)
    noisy_img = noisy_img.cuda()

    out_img = model(noisy_img)
    # print(out_img.shape)
    # print(clean_img.shape)

    out_img = out_img.cpu()
    out_img = out_img.detach()
    out_img = torch.clamp(out_img, 0., 1.)
    out_img = torch.squeeze(out_img)
    save_img = out_img * 255
    save_img = torch.round(save_img)
    save_img = save_img.permute(1,2,0)

    # compute PSNR
    out = out_img.numpy().astype(np.float32)
    clean = clean_img.numpy().astype(np.float32)
    PSNR = peak_signal_noise_ratio(clean, out, data_range=1.)
    total_PSNR += PSNR
    if PSNR > max_PSNR:
        max_PSNR = PSNR
        max_name = name
    if PSNR < min_PSNR:
        min_PSNR = PSNR
        min_name = name

    # save image
    io.imsave(save_dir+'/%s_%0.4f.png' % (name, PSNR), save_img)

print('Total PSNR of sigma=%d: %f' % (args.noise_sigma, total_PSNR/len(test_data)))
print('Max PSNR: %s %f' % (max_name, max_PSNR))
print('Min PSNR: %s %f' % (min_name, min_PSNR))

