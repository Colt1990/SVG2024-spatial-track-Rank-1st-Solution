import os
import time
import numpy as np
import torch
from einops import rearrange, repeat

from models.mm_diffusion import dist_util, logger
from models.mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from models.mm_diffusion.script_util import (
    image_sr_model_and_diffusion_defaults,
    image_sr_create_model_and_diffusion
)

from models.mm_diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from models.mm_diffusion.dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver


class SpatialTrackBaselineModel:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dist_util.setup_dist("G8")

        diffusion_config = {
            'video_size': [20,3,64,64],
            'audio_size': [2,80000],
            'num_channels': 128,
            'num_res_blocks': 2,
            'num_heads': 4,
            'num_heads_upsample': -1,
            'num_head_channels': 64,
            'cross_attention_resolutions': '2,4,8',
            'cross_attention_windows': '1,4,8',
            'cross_attention_shift': True,
            'video_attention_resolutions': '2,4,8',
            'audio_attention_resolutions': '-1',
            'channel_mult': '',
            'dropout': 0.0,
            'class_cond': False,
            'use_checkpoint': False,
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            'use_fp16': True,
            'video_type': '2d+1d',
            'audio_type': '1d',
            'learn_sigma': False,
            'diffusion_steps': 1100,
            'noise_schedule': 'linear',
            # 'noise_schedule': 'cosine',
            'timestep_respacing': '',
            'use_kl': False,
            'predict_xstart': False,
            'rescale_timesteps': False,
            'rescale_learned_sigmas': False,
             # 'sample_fn':'ddpm'
        }

        sr_config = {
            'sr_num_channels': 192,
            'sr_num_res_blocks': 2,
            'sr_num_heads': 4,
            'sr_num_heads_upsample': -1,
            'sr_num_head_channels': -1,
            'sr_attention_resolutions': '8,16,32',
            'sr_dropout': 0.0,
            'sr_class_cond': False,
            'use_checkpoint': False,
            'sr_use_scale_shift_norm': True,
            'sr_resblock_updown': True,
            'use_fp16': True,
            'noise_schedule': 'linear', #linear
            'use_kl': False,
            'predict_xstart': False,
            'rescale_timesteps': False,
            'rescale_learned_sigmas': False,
            'sr_learn_sigma': True,
            'large_size': 256,
            'small_size': 64,
            'sr_diffusion_steps': 840,
            'sr_timestep_respacing': 'ddim25'
        }

        self.multimodal_model, self.multimodal_diffusion = create_model_and_diffusion(**diffusion_config)
        self.sr_model, self.sr_diffusion = image_sr_create_model_and_diffusion(**sr_config)

        self.sr_model.load_state_dict_(
            dist_util.load_state_dict('models/pretrained_models/model070000.pt', map_location="cpu"), is_strict=True
        )    #model_SR_mmdiff_120000.pt
        self.sr_model.to(self.device)
        self.sr_model.convert_to_fp16()
        self.sr_model.eval()

        self.multimodal_model.load_state_dict(
            dist_util.load_state_dict('models/pretrained_models/model330011.pt', map_location="cpu")
        )
        self.multimodal_model.to(self.device)
        self.multimodal_model.convert_to_fp16()
        self.multimodal_model.eval()

        self.video_size = diffusion_config['video_size']
        self.audio_size = diffusion_config['audio_size']
        self.video_sr_size = sr_config['large_size']

        self.sampling_steps = 80


    def get_batch_size(self): # Will only be called once at the start of evaluation
        # Change the batch size that will fit in a 24 GB A10G GPU
        self.batch_size = 1
        return self.batch_size
    
    def generate_videos(self, seed_list): 
        """ 
        Generate videos for a batch of seeds (list of ints)
        Note: Size of batch may be equal to or less than the batch size requested (for the last batch it can be smaller than batch size)
        
        Return a batch of audios and video arrays that should be savabale using 

            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            from moviepy.audio.AudioClip import AudioArrayClip
            audio_clip = AudioArrayClip(audio, fps=audio_fps)
            video_clip = ImageSequenceClip(imgs, fps=video_fps)
            video_clip = video_clip.with_audio(audio_clip)
            video_clip.write_videofile(output_path, video_fps, audio=True, audio_fps=audio_fps)

        """

        shape = {
            "video": (self.batch_size , *self.video_size), \
            "audio": (self.batch_size , *self.audio_size)
        }
        
        start = time.time()

        dpm_solver = multimodal_DPM_Solver(model=self.multimodal_model,
                                           alphas_cumprod=torch.tensor(self.multimodal_diffusion.alphas_cumprod, 
                                                                       dtype=torch.float32),
                                           
                                           predict_x0=True, 
                                           thresholding=True,
                                          max_val = 1.2
                                          )
        
        x_T = {
            "video": torch.randn(shape["video"]).to(dist_util.dev()),
            "audio": torch.randn(shape["audio"]).to(dist_util.dev())
        }
        
        # sample = dpm_solver.sample(
        #     x_T,
        #     steps=80,
        #     order=2,
        #     skip_type="logSNR", #"logSNR", #
        #     method="singlestep",
        #     # t_end = 1/1100,
        #     # t_start=0.95
        # )
        sample = dpm_solver.sample(
            x_T,
            steps=80,
            order=2,
            skip_type="logSNR", #"logSNR", #
            method="singlestep",
            # t_end = 1/1100,
            # t_start=0.95
        )
        # sample = dpm_solver.progressive_sample(
        #     x_T,
        #     steps=80,
        #     order=2,
        #     skip_type="logSNR", #"logSNR", #
        #     method="singlestep",
        #     t_end = 1/1000,
        #     t_start=1
        # )

        # sample = dpm_solver.progressive_sample_with_resampling(
        #     x_T,
        #     steps=80,
        #     order=2,
        #     skip_type="logSNR", #"logSNR", #
        #     method="singlestep",
        #     t_end = 1/1000,
        #     t_start=1
        # )

    
        
          # 较早开始应用空间对齐

        # sample = dpm_solver.sample(
        #     x_T,
        #     steps=80,
        #     order=2,
        #     skip_type="logSNR", #"logSNR", #
        #     method="singlestep",
        #     # t_end = 1/1100,
        #     # t_start=0.9
        # )

        # sample = dpm_solver.sample(
        #     x_T,
        #     steps=self.sampling_steps,
        #     order=2,
        #     skip_type="time_uniform", #"logSNR", #
        #     method="adaptive",
        # )


     

    #     sample = dpm_solver.sample(
    #         x_T,
    #         steps=80,
    #         order=3,
    #         skip_type="time_uniform", #"logSNR", #
    #         method="adaptive",
    #         # t_end=1e-4,
    #         # t_start=1.0,
    # # t_end=1e-3,
    # # denoise=True
    
    # #         atol=0.001,           # 更严格容差
    # # rtol=0.01,            # 更严格容差
    # # denoise=True,         # 最后去噪
    #     )
            
        

        # sample = dpm_solver.sample(
        #     x_T,
        #     steps=80,
        #     # steps=80,
        #     order=2,
        #     skip_type="logSNR", #"logSNR", #
        #     method="singlestep",
        #     # t_start=0.9
        # )


        end = time.time()
        print("Sample time", end - start)    

        video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        audio = sample["audio"]              
        video = video.permute(0, 1, 3, 4, 2)
        video = video.contiguous()

        all_audios = audio.detach().cpu().numpy()

        model_kwargs = {'low_res': sample["video"]}

        ### 对比度调节
        # from torchvision.transforms import functional as TF
        # def preprocess_low_res(video):
        #     video = (video + 1) / 2.0  # [-1, 1] -> [0, 1]
        #     video = TF.adjust_contrast(video, contrast_factor=1.2)  # 增强对比度
        #     return video * 2.0 - 1.0  # 回到 [-1, 1]
        # model_kwargs['low_res'] = preprocess_low_res(model_kwargs['low_res'])
                
        
        
        b, f, c, h, w = sample["video"].shape
        shape = (b*f, c, self.video_sr_size, self.video_sr_size)
        model_kwargs['low_res'] = rearrange(model_kwargs['low_res'], 'b f c h w -> (b f) c h w')
        noise = torch.randn((b, c, self.video_sr_size, self.video_sr_size)).to(dist_util.dev())
        noise = repeat(noise, 'b c h w -> (b repeat) c h w', repeat=f)

        sample_fn = self.sr_diffusion.ddim_sample_loop
   
        
        sr_sample = sample_fn(
            self.sr_model,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            noise=noise,
            # eta=0.0  # 确定性采样
        )


        

        end = time.time()
        print("SR Time: ", end - start)  

        #平滑操作
     
        

        sr_sample = ((sr_sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        

        
        sr_sample = sr_sample.permute(0, 2, 3, 1)
        sr_sample = rearrange(sr_sample, '(b f) h w c-> b f h w c', b=self.batch_size)
        video_sr_samples = sr_sample.contiguous().cpu().numpy()               
                
        output = {}
        output['audios'] = list(all_audios)
        output['videos'] = list(video_sr_samples)
        return output
