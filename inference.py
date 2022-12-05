
import os
import os.path as P
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
from model import Regnet
from config import _C as config
from tqdm import tqdm
from test import build_wavenet, gen_waveform
from extract_rgb_flow import cal_for_frames
from extract_feature import GroupNormalize, GroupScale, Stack, ToTorchFormatTensor


import pickle as pkl
import torch.nn.parallel
from PIL import Image
import torch.optim
import torchvision
from glob import glob
from tsn.models import TSN


def init_model():
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    model = Regnet().netG  # only needs Generator when inference
    
    # model = model.module    
    if config.checkpoint_path != '':
        state_dict = torch.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict["optimizer_netG"])
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wavenet_model = build_wavenet(config.wavenet_path, device)
    return model, wavenet_model


def extract_feature_one_video(rgb_flow_dir, modality):
    
    net = TSN(modality, consensus_type='avg', dropout=0.7)

    cropping = torchvision.transforms.Compose([
        GroupScale((net.input_size, net.input_size)),
    ])
    
    transform = torchvision.transforms.Compose([
        cropping, Stack(roll=True),
        ToTorchFormatTensor(div=False),
        GroupNormalize(net.input_mean, net.input_std),])

    images = []
    if modality == 'RGB':
        files = glob(P.join(rgb_flow_dir, "img*.jpg"))
    else:
        files = glob(P.join(rgb_flow_dir, "flow_x*.jpg"))
    files.sort()
    for file in files:
        if modality == 'RGB':
            images.extend([Image.open(file).convert('RGB')])
        if modality == 'Flow':
            x_img = Image.open(file).convert('L')
            y_img = Image.open(file.replace("flow_x*.jpg", "flow_y*.jpg")).convert('L')
            images.extend([x_img, y_img])
    data = transform(images)
        
    length = 3 if modality == 'RGB' else 2    
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(1), data.size(2)), volatile=True)
    rst = np.squeeze(net(input_var).data.cpu().numpy().copy())
    return rst


def inference_for_one_video(model, wavenet, video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    video_path_fps = P.join(save_dir, Path(video_path).stem+f'_21.5fps.mp4')
    os.system(f'ffmpeg -y -i {video_path} -loglevel error -r 21.5 -c:v libx264 -strict -2 -y {video_path_fps}')
    
    output_dir = P.join(save_dir, 'rgb_flow')
    os.makedirs(output_dir, exist_ok=True)
    print('start extract rgb and optical flow')
    cal_for_frames(video_path_fps, output_dir=output_dir, width=340, height=256, infer=True)
    print('rgb and optical flow saved')
    rgb_feature = extract_feature_one_video(output_dir, 'RGB')
    flow_feature = extract_feature_one_video(output_dir, 'Flow')
    pkl.dump(rgb_feature, open(P.join(save_dir, 'rgb.pkl'), "wb"))
    pkl.dump(flow_feature, open(P.join(save_dir, 'flow.pkl'), "wb"))
    print('rgb feature and flow feature saved')

    #rgb_feature = pkl.load(open(P.join(save_dir, 'rgb.pkl'), "rb"))
    #flow_feature = pkl.load(open(P.join(save_dir, 'flow.pkl'), "rb"))  # use pre-computed feature 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        inputs = torch.cat([torch.tensor(rgb_feature), torch.tensor(flow_feature)], 1).unsqueeze(0)
                
        encoder_output = model.encoder(inputs.to(device))
        
        # use zero when inference
        auxiliary = torch.zeros(1, encoder_output.shape[1], config.auxiliary_dim).to(device)
        encoder_output = torch.cat([encoder_output, auxiliary], dim=2)
        mel_output_decoder = model.decoder(encoder_output)
        mel_output_postnet = model.postnet(mel_output_decoder)
        mel_output = mel_output_decoder + mel_output_postnet
        mel_spec = mel_output[0].data.cpu().numpy()
        save_path = P.join(config.save_dir, Path(video_path).stem+".wav")    

        # gen_waveform(wavenet, save_path, mel_spec[:, :200], device) # for part audio
        gen_waveform(wavenet, save_path, mel_spec, device)   # for whole audio 
        print(f'audio saved to {P.abspath(save_path)}')
        result_path = video_path_fps.replace('_21.5fps.mp4', '_result.mp4')        
        os.system(f"ffmpeg -loglevel panic -i {video_path_fps} -i {save_path} -c:v copy -c:a aac -strict experimental -y {result_path}")
        print(f'result saved to {P.abspath(result_path)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--video_path', type=str, default='')    
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')    
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()    

    if args.config_file:
        config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)

    model, wavenet = init_model()
    video_path = args.video_path
    save_dir = P.join(config.save_dir, Path(video_path).stem)
    inference_for_one_video(model, wavenet, video_path, save_dir)