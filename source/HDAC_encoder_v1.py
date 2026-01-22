import os
import time
import torch
import imageio
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.HDAC_utils import *
from GFVC.HDAC.metric_utils import eval_metrics
from GFVC.HDAC.bl_codecs import bl_encoders
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List
from PIL import Image


class HDAC:
    '''HDAC (Hybrid Deep Animation Coding) 编码器类
    实现了基于深度学习的视频编码框架，支持混合编码（base layer + 运动表示）
    '''
    def __init__(self, hdac_config_path:str, hdac_checkpoint_path:str,adaptive_metric='lpips',adaptive_thresh=0.5, device='cpu') -> None:
        '''
        初始化HDAC编码器
        
        Args:
            hdac_config_path: HDAC模型配置文件路径
            hdac_checkpoint_path: HDAC模型权重文件路径
            adaptive_metric: 自适应参考帧选择的质量评估指标 (lpips/dists/psnr/ssim)
            adaptive_thresh: 自适应参考帧选择的阈值
            device: 执行设备 (cpu/cuda)
        '''
        self.device = device
        # 加载HDAC模型（分析网络和合成网络）
        dac_analysis_model, dac_synthesis_model = load_hdac_checkpoints(hdac_config_path, hdac_checkpoint_path, device=device)
        self.analysis_model = dac_analysis_model.to(device)    # 分析网络：提取运动表示
        self.synthesis_model = dac_synthesis_model.to(device)  # 合成网络：重建视频帧

        self.base_layer = None  # 基础层视频（用于混合编码）
        self.reference_frame = None  # 当前参考帧
        self.kp_reference = None     # 当前参考帧的运动关键点

        # 自适应刷新工具：用于选择最佳参考帧
        self.adaptive_metric = adaptive_metric.lower()
        if self.adaptive_metric in ['lpips', 'dists']:
            # 需要设备的评估指标
            self.metric = eval_metrics[self.adaptive_metric](device=self.device)
        else:
            # 不需要设备的评估指标
            self.metric = eval_metrics[self.adaptive_metric]()
        self.thresh = adaptive_thresh  # 自适应阈值（需与选择的指标匹配）

        self.avg_quality = AverageMeter()  # 质量指标平均值计算器

    def predict_inter_frame(self,kp_inter_frame: Dict[str,torch.Tensor],frame_idx:int)->torch.Tensor:
        '''
        预测中间帧（与GFVC解码器的预测过程一致）
        
        Args:
            kp_inter_frame: 中间帧的运动关键点表示
            frame_idx: 帧索引
            
        Returns:
            torch.Tensor: 预测的视频帧
        '''
        # 如果使用基础层编码，获取对应帧的基础层数据
        if self.base_layer is not None:
            base_layer_frame = frame2tensor(self.base_layer[frame_idx]).to(self.device)
        else:
            base_layer_frame = None

        # 使用合成网络预测中间帧
        prediction = self.synthesis_model.predict(
            self.reference_frame,    # 参考帧
            self.kp_reference,       # 参考帧的运动关键点
            kp_inter_frame,          # 当前帧的运动关键点
            base_layer_frame         # 基础层帧（可选）
        )
        return prediction


def resize_frames(frames: np.ndarray, scale_factor=1)->np.ndarray:
    N, H, W, C = frames.shape
    if scale_factor != 1:
        out = []
        for idx in range(N):
            img = Image.fromarray(frames[idx])
            img = img.resize((int(H*scale_factor), int(W*scale_factor)),resample=Image.Resampling.LANCZOS)
            out.append(np.asarray(img))
        return np.array(out)
    else:
        return frames
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--ref_codec", default='vtm', type=str,help="Reference frame coder [vtm | lic]")
    parser.add_argument("--adaptive_metric", default='PSNR', type=str,help="RD adaptation metric (for selecting reference frames to keep in buffer)")
    parser.add_argument("--adaptive_thresh", default=30, type=float,help="Reference selection threshold")
    parser.add_argument("--use_base_layer", default='ON', type=str,help="Flag to use hybrid coding framework (OFF sets animation-only reconstruction)")
    parser.add_argument("--base_codec", default='hevc', type=str,help="Base layer codec [hevc | vvc]")
    parser.add_argument("--bl_qp", default=50, type=int,help="QP value for encoding the base layer")
    parser.add_argument("--bl_scale_factor", default=1.0, type=float,help="subsampling factor for base layer frames")
    parser.add_argument("--num_kp", default=10, type=int,help="Number of motion keypoints")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=int(opt.seq_width)
    height=int(opt.seq_height)
    q_step= opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    gop_size = opt.gop_size
    device = opt.device
    thresh = int(opt.adaptive_thresh)
    num_kp = int(opt.num_kp)

    #base layer params
    use_base_layer = opt.use_base_layer
    bl_codec_name = opt.base_codec
    bl_qp = opt.bl_qp
    bl_scale_factor = opt.bl_scale_factor
    ###

    if not torch.cuda.is_available():
        device = 'cpu'
    

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    
    model_name = 'HDAC' 
    model_dirname=f'../experiment/{model_name}_{bl_codec_name.upper()}/Iframe_'+str(iframe_format)   
    
    

    ###################
    start=time.time()
    kp_output_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/'    #OutPut directory for motion keypoints
    os.makedirs(kp_output_path,exist_ok=True)                   

    enc_output_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/' ## OutPut path for encoded reference frame     
    os.makedirs(enc_output_path,exist_ok=True)     

    bl_output_path =model_dirname+'/bl/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp) ## OutPut path for encoded HEVC| VVC base layer  
    os.makedirs(bl_output_path,exist_ok=True)                     

    # 根据文件格式选择正确的读取函数
    if original_seq.endswith('.yuv') or original_seq.endswith('.YUV'):
        # 对于YUV420格式文件，使用yuv420_to_rgb444函数
        from GFVC.utils import yuv420_to_rgb444
        rgb_frames = yuv420_to_rgb444(original_seq, width, height, 0, frames, show=False, out=False)
        # 将RGB帧转换为listR, listG, listB格式
        listR = []
        listG = []
        listB = []
        for frame in rgb_frames:
            R, G, B = frame[:,:,0], frame[:,:,1], frame[:,:,2]
            listR.append(R)
            listG.append(G)
            listB.append(B)
    else:
        # 对于RGB444格式文件，使用raw_reader_planar函数
        listR,listG,listB=raw_reader_planar(original_seq,width, height,frames)

    seq_kp_integer = []
    sum_bits = 0

    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name=opt.ref_codec, device=device)
    #Motion keypoints coding
    kp_coder = KPEncoder(kp_output_path,q_step,num_kp=num_kp, device=device)

    

    #Main encoder models wrapper
    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path= f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    enc_main = HDAC(model_config_path, model_checkpoint_path,
                   opt.adaptive_metric,opt.adaptive_thresh,device=device)

    
    # 主编码循环，使用torch.no_grad()禁用梯度计算以提高性能
    with torch.no_grad():
        out_video = []  # 输出视频帧列表
        g_start, g_end = 0, 0  # GOP（图像组）的起始和结束索引
        
        # 处理基础层（如果启用）
        if use_base_layer == 'ON':
            # 创建覆盖整个序列的基础层
            print(f'创建 {bl_codec_name} 基础层..')
            # 将RGB通道转换为视频序列格式 (N, H, W, C)
            gop = np.transpose(np.array([listR, listG, listB]), [1, 2, 3, 0])
            N, h, w, _ = gop.shape
            
            # 基础层编码参数
            bl_coding_params = {
                    'qp': bl_qp,                    # 编码质量参数
                    'sequence': resize_frames(gop, bl_scale_factor),  # 缩放后的视频序列
                    'fps': 25,                      # 帧率
                    'frame_dim': (h, w),            # 帧维度
                    'log_path': bl_output_path      # 输出路径
                }
            
            # 创建并运行基础层编码器
            bl_enc = bl_encoders[bl_codec_name](**bl_coding_params)
            info_out = bl_enc.run()
            
            # 处理编码后的基础层帧
            bl_frames = np.transpose(
                resize_frames(info_out['dec_frames'], 1//bl_scale_factor),
                [0, 3, 1, 2]  # 转换为 (N, 3, H, W) 格式
            )
            
            sum_bits += info_out['bits']  # 累计基础层比特数
            enc_main.base_layer = bl_frames  # 设置基础层数据
    
        # 提取动画关键点并进行编码
        print("Extracting animation keypoints..")    
        
        # 遍历所有帧进行编码
        for frame_idx in tqdm(range(0, frames)):            
            # 获取当前帧的RGB通道数据
            current_frame = [listR[frame_idx], listG[frame_idx], listB[frame_idx]]
            cur_out = np.transpose(np.array(current_frame), [1, 2, 0])  # 转换为 (H, W, 3) 格式
            
            if frame_idx == 0:  # 处理I帧（关键帧）
                # 使用参考编码器压缩I帧
                reference, ref_bits = ref_coder.compress(current_frame, frame_idx)
                sum_bits += ref_bits  # 累计比特数
                
                # 处理不同编码器返回的参考帧格式
                if isinstance(reference, np.ndarray):
                    # 如果是numpy数组，转换为tensor格式
                    out_fr = reference
                    # 转换格式: HxWx3 (uint8) -> 1x3xHxW (float32)
                    reference = frame2tensor(np.transpose(reference, [2, 0, 1]))
                else:
                    # 如果是tensor格式（如使用LIC编码器），直接转换为numpy用于输出
                    out_fr = np.transpose(tensor2frame(reference), [1, 2, 0])
                
                # 生成输出视频帧（原始帧 + 基础层帧 + 重建帧）
                if use_base_layer == 'ON':
                    bl_fr = np.transpose(enc_main.base_layer[frame_idx], [1, 2, 0])
                    out_video.append(np.concatenate([cur_out, bl_fr, out_fr], axis=1))
                else:
                    out_video.append(np.concatenate([cur_out, out_fr], axis=1))
                    
                # 将参考帧转移到指定设备
                reference = reference.to(device)
                
                # 提取运动表示向量（关键点、紧凑特征等）
                kp_reference = enc_main.analysis_model(reference)
                kp_value_frame = kp_coder.get_kp_list(kp_reference, frame_idx)
                
                # 将关键点表示添加到列表，用于预测编码下一帧的关键点
                kp_coder.rec_sem.append(kp_value_frame)
                kp_coder.ref_frame_idx.append(frame_idx)  # 参考帧索引的元数据
                
                # 更新编码器的参考帧信息
                enc_main.reference_frame = reference
                enc_main.kp_reference = kp_reference
            else:  # 处理P帧（预测帧）
                # 合并RGB通道并调整大小
                inter_frame = cv2.merge(current_frame)
                inter_frame = resize(inter_frame, (width, height))[..., :3]
                
                # 转换为tensor格式并转移到设备
                inter_frame = torch.tensor(inter_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                inter_frame = inter_frame.to(device)  # 使用指定设备
                
                # 提取当前帧的运动表示
                kp_inter_frame = enc_main.analysis_model(inter_frame)
                kp_frame = kp_coder.get_kp_list(kp_inter_frame, frame_idx)
                
                # 编码运动关键点
                rec_kp_frame, kp_bits = kp_coder.encode_kp(kp_frame, frame_idx)
                sum_bits += kp_bits  # 累计关键点比特数
                
                # 重建当前帧并评估质量
                pred = enc_main.predict_inter_frame(rec_kp_frame, frame_idx)
                pred = np.transpose(tensor2frame(pred), [1, 2, 0])  # 转换为显示格式
                
                # 生成输出视频帧
                if use_base_layer == 'ON':
                    bl_fr = np.transpose(enc_main.base_layer[frame_idx], [1, 2, 0])
                    out_video.append(np.concatenate([cur_out, bl_fr, pred], axis=1))
                else:
                    out_video.append(np.concatenate([cur_out, pred], axis=1))
               
        # 保存输出视频（包含原始帧、基础层帧和重建帧）
        imageio.mimsave(f"{enc_output_path}enc_video.mp4", out_video, fps=25.0)
        
        # 编码元数据并累计比特数
        sum_bits += kp_coder.encode_metadata()
        
        # 计算并输出总编码时间和总比特数
        end = time.time()
        print(f"关键点提取成功。总耗时: {end-start:.4f}秒。关键点编码总比特数: {sum_bits} bits。")






