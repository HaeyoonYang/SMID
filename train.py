import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from preprocessing import *
from EDSR import *
from utils import weights_init_kaiming, batch_psnr
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics.simple_metrics import peak_signal_noise_ratio


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--model', default='EDSR_base', type=str, help='choose a type of model')
parser.add_argument('--train_iter', default=1, type=int, help='(train_iter)th training')
parser.add_argument('--patch_size', default=128, type=int, help='patch size')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--data_path', default='./data', type=str, help='path of data')
parser.add_argument('--log_dir', default='./logs', type=str, help='path of log')
parser.add_argument('--noiseInt', default=[0, 75], type=int, help="training noise interval")
parser.add_argument('--valNoise', default=50, type=int, help="noise level on validation set")
parser.add_argument('--epoch', default=700, type=int, help='number of train epoches')
parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--milestone', default=150, type=int, help='milestones of scheduler')
args = parser.parse_args()


cuda = torch.cuda.is_available()
save_dir = './models'
log_dir = os.path.join(args.log_dir, str(args.train_iter))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter()


if __name__ == "__main__":
    print('>Loading Dataset ...')
    train_data = Data(data_path=args.data_path, train=True)
    valid_data = ValidData(data_path=args.data_path, sigma=args.valNoise)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)

    model = EDSR()
    # model.apply(weights_init_kaiming)
    criterion = nn.L1Loss(size_average=False)
    if cuda:
        model.cuda()
        criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.milestone, gamma=0.5)

    for epoch in range(args.epoch):
        epoch_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            clean_imgs, noisy_imgs = batch
            clean_imgs, noisy_imgs = clean_imgs.cuda(), noisy_imgs.cuda()

            out_imgs = model(noisy_imgs)
            loss = criterion(out_imgs, clean_imgs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                output = torch.clamp(out_imgs, 0., 1.)
                psnr_train = batch_psnr(output, clean_imgs, 1.)
                print('[epoch %d] [%d / %d] loss = %.4f PSNR: %.4f' % (epoch+1, i+1, len(train_loader), loss.item(), psnr_train))


        model.eval()
        psnr_valid = 0
        valid_loss = 0
        for _, data in enumerate(valid_data):
            clean_img, noisy_img = data
            noisy_img = torch.unsqueeze(noisy_img, 0)
            clean_img, noisy_img = clean_img.cuda(), noisy_img.cuda()
            out_img = model(noisy_img)
            out_img = torch.squeeze(out_img)
            loss = criterion(out_img, clean_img)
            valid_loss += loss.item()

            out_img = out_img.cpu().detach()
            clean_img = clean_img.cpu().detach()
            out_img = torch.clamp(out_img, 0., 1.)

            # compute PSNR
            out = out_img.numpy().astype(np.float32)
            clean = clean_img.numpy().astype(np.float32)
            PSNR = peak_signal_noise_ratio(clean, out, data_range=1.)
            psnr_valid += PSNR

        psnr_valid /= len(valid_data)
        valid_loss /= len(valid_data)
        print("[epoch %d] Validation Loss: %.4f, PSNR: %.4f" % (epoch+1, valid_loss, psnr_valid))

        writer.add_scalar('train loss', epoch_loss, epoch)
        writer.add_scalar('valid loss', valid_loss, epoch)
        writer.add_scalar('valid psnr', psnr_valid, epoch)

        if epoch % 10 == 9:
            save_dict = {'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(log_dir, str(epoch+1)+'_ckpt.pth'))

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_00'+str(args.train_iter)+'.pth'))

