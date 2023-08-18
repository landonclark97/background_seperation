import numpy as np

import pickle

import cv2

import matplotlib.pyplot as plt

import glob
from PIL import Image



H = 150
W = 200



vid_name = 'two_persons_walking'
# vid_name = 'robot_reach'
# vid_name = 'human_rob_int'

# method='ircur'
# method='st_pcp'
# method='pcp'
method='admm'
# method = 'frpcag'
# method = 'pcagtv'

SCALE = 1.0

MATLAB = False
if (method == 'frpcag') or (method == 'pcagtv'):
    MATLAB = True
if method == 'admm':
    SCALE = 255.0



def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100.0
    max_pixel = 1.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def TP(pred, true):
    return np.where(pred[np.where(true > 0.0)] > 0.0)[0].shape[0]

def FP(pred, true):
    return np.where(pred[np.where(true == 0.0)] > 0.0)[0].shape[0]

def TN(pred, true):
    return np.where(pred[np.where(true == 0.0)] == 0.0)[0].shape[0]

def FN(pred, true):
    return np.where(pred[np.where(true > 0.0)] == 0.0)[0].shape[0]


if MATLAB:
    with open('./L/' + method + '_' + vid_name + '_L.pkl', 'rb') as f:
        L = pickle.load(f)
    with open('./S/' + method + '_' + vid_name + '_S.pkl', 'rb') as f:
        S = pickle.load(f)
    L = np.swapaxes(np.reshape(L,(-1,W,H)),0,2)
    S = np.swapaxes(np.reshape(S,(-1,W,H)),0,2)
else:
    L = np.load('./L/' + method + '_' + vid_name + '_L.npy')
    S = np.load('./S/' + method + '_' + vid_name + '_S.npy')

# L = L[:,:,1:]
# S = S[:,:,1:]


L /= SCALE
S /= SCALE
# L /= np.amax(L)
# L = (L-np.amin(L))/(np.amax(L)-np.amin(L))
# S /= np.amax(S)
# S = (S-np.amin(S))/(np.amax(S)-np.amin(S))

print(np.amax(L))
print(np.amin(L))

print(L.shape)


# two person walking crop
# if vid_name == 'two_persons_walking':
#     crop_left = 0
#     crop_top = 0
#     crop_right = 0
#     crop_bottom = 0

# # robot reach crop
# elif vid_name == 'robot_reach':
#     crop_left = 15
#     crop_top = 500
#     crop_right = 0
#     crop_bottom = 300

# # human robot interaction
# elif vid_name == 'human_rob_int':
#     crop_left = 20
#     crop_top = 507
#     crop_right = 0
#     crop_bottom = 320



b_name = '../../data/'+vid_name+'_motionless/b_img.png'

b_img = cv2.imread(b_name)
b_img = cv2.resize(b_img.mean(-1), (W,H), None, None)
b_img /= 255.0
# b_img /= np.amax(b_img)


f_name = '../../data/'+vid_name+'_motionless_mask/'
f_im_list = glob.glob(f_name+'frame*.png')

frames = len(f_im_list)
# if L.shape[2] >= frames:
    # L = L[:,:,L.shape[2]-frames:]
    # S = S[:,:,S.shape[2]-frames:]
# else:
    # frames = L.shape[2]

f_mask = np.empty((H,W,frames))
for f in range(1,frames+1):
    f_im = cv2.imread(f_name+'frame'+str(f)+'.png')
    im = cv2.resize(f_im.mean(-1), (W,H), None, None)
    f_mask[:,:,f-1] = im

f_mask /= np.amax(f_mask)


print(f_mask.shape)


L_mean = np.mean(L,axis=2)


b_img_mx = np.amax(b_img)
b_img_mn = np.amin(b_img)
L_mx = np.amax(L_mean)
L_mn = np.amin(L_mean)


print(b_img_mx, b_img_mn, L_mx, L_mn)

# b_img = (b_img-b_img_mn)/(b_img_mx-b_img_mn)
# L_mean = (L_mean-L_mn)/(L_mx-L_mn)
# print('PSNR: ', PSNR((b_img-b_img_mn)*(b_img_mx-b_img_mn),(L_mean-L_mn)*(L_mx-L_mn)))
print('PSNR:', PSNR(b_img, L_mean))
print('RE:',(np.linalg.norm(L_mean-b_img,ord='fro'))/np.linalg.norm(b_img,ord='fro'))


fig = plt.figure()
axs = fig.add_subplot(1,3,1)
# axs2 = fig.add_subplot(1,3,2)
# axs3 = fig.add_subplot(1,3,3)

axs.imshow(L_mean,cmap='gray',vmin=0.0,vmax=1.0)
# axs2.imshow(b_img,cmap='gray')
# axs3.imshow(L_mean-b_img,cmap='gray',vmin=-1.0,vmax=1.0)
plt.show()


threshs = np.linspace(0.0,1.0,400)

S_vec = np.abs(S.flatten())
F_vec = f_mask.flatten()

assert S_vec.shape[0] == F_vec.shape[0]

max_res = 0.0
res = []

tprs = []
fprs = []

for th in threshs:

    t_S_vec = np.where(S_vec >= th, 1.0, 0.0)

    tp = TP(t_S_vec, F_vec)
    fp = FP(t_S_vec, F_vec)
    tn = TN(t_S_vec, F_vec)
    fn = FN(t_S_vec, F_vec)

    tprs.append(float(tp)/float(tp+fn))
    fprs.append(float(fp)/float(fp+tn))

    try:
        pr = float(tp)/float(tp+fp)
        re = float(tp)/float(tp+fn)
    except:
        continue

    try:
        fm = 1.0/((1.0/(2.0*pr))+(1.0/(2.0*re)))
    except:
        continue

    if pr*re*fm > max_res:
    # if pr+re+fm > max_res:
        # max_res = pr+re+fm
        max_res = pr*re*fm
        ideal_th = th
        res = [pr, re, fm, th]


print('Precision: ', res[0])
print('Recall: ', res[1])
print('F-measure: ', res[2])
print('Optimal Threshold: ', res[3])


np_tprs = np.array(tprs)
np_fprs = np.array(fprs)

s_ind = np.argsort(np_fprs)

s_tprs = np_tprs[s_ind]
s_fprs = np_fprs[s_ind]


auc = 0.0
for i in range(s_ind.shape[0]-1):
    auc += (s_fprs[i+1]-s_fprs[i])*(0.5*(s_tprs[i]+s_tprs[i+1]))

print('AUC: ', auc)


ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(L_mean, cmap='gray',vmin=0.0, vmax=1.0)

    background = L[:,:, ind]#.reshape(small_frames[ind].shape)
    foreground = S[:,:, ind]#.reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray',vmin=0.0, vmax=1.0)
    axs[1, i].set_title(method.upper() + ", L")
    axs[2, i].imshow(foreground, cmap='gray', vmin=-1.0, vmax=1.0)
    axs[2, i].set_title(method.upper() + ", S")


fig.tight_layout()
plt.savefig("./res/" + method + "/" + method + "_" + vid_name + ".pdf")


fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(s_fprs, s_tprs)
ax.set_title('ROC')
ax.set_xlabel('FP Rate')
ax.set_ylabel('TP Rate')
plt.savefig("./res/" + method + "/" + method + "_" + vid_name + "_ROC.pdf")
