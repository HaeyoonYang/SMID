
import glob
import numpy as np
import shutil
from skimage import io
import torch


# img_dirs = sorted(glob.glob('results/50/*'))
# comp_dirs = sorted(glob.glob('compResults/50/*'))
#
#
# img_psnr = []
# comp_psnr = []
# name = []
# for i in range(len(img_dirs)):
#     img_dir = img_dirs[i]
#     comp_dir = comp_dirs[i]
#     img_name = img_dir[11:-12]
#     img_psnr.append(float(img_dir[-11:-4]))
#     comp_psnr.append(float(comp_dir[-11:-4]))
#     name.append(img_name)
#
# img = np.array(img_psnr)
# comp = np.array(comp_psnr)
# name = np.array(name)
#
# sub = img - comp
#
# # print(img[0:5])
# # print(comp[0:5])
# # print(sub[0:5])
# # print(-sub[0:5])
#
# sorted_sub = np.argsort(-sub)
# sorted_name = name[sorted_sub]
#
# print(sorted_name[0:10])
# print(sub[sorted_sub][0:10])
#
#
# # print(len(img_dir))
# #
# # for i in range(len(img_dir)):
# #     if i % 33 == 0:
# #         shutil.move(img_dir[i], valid_dir+'/img')
# #         shutil.move(seg_dir[i], valid_dir+'/seg')
# #     else:
# #         shutil.move(img_dir[i], train_dir+'/img')
# #         shutil.move(seg_dir[i], train_dir+'/seg')



image_dir = 'data/CBSD/101085.png'
img = io.imread(image_dir)

clean = img.transpose((2,0,1))
clean = torch.tensor(clean).float()/255.0

sigma = 50/255.0
add_noise = torch.FloatTensor(clean.size()).normal_(mean=0, std=sigma)
noisy_img = clean + add_noise

# Save noisy image
save_img = torch.clamp(noisy_img, 0., 1.)
save_img = save_img * 255
save_img = torch.round(save_img)
save_img = save_img.permute(1,2,0)

io.imsave('101085_50.png', save_img)
