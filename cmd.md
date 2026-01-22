

# 2026.1.12
python HDAC_encoder_v1.py  --original_seq E:/HWpro2026/dataset/train/input011218.yuv --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

python HDAC_encoder_v1.py  --original_seq E:/HWpro2026/dataset/train/input011218.yuv --encoding_frames 400 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 400

python HDAC_encoder_v1_260112.py  --original_seq E:/HWpro2026/dataset/train/input0113.mp4 --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 E:/HWpro2026/dataset/face/face_20260112.mp4

python HDAC_encoder_v1_260112.py  --original_seq E:/HWpro2026/dataset/face/face_20260112.mp4 --encoding_frames 113 --gop_size 113 --seq_width 1920 --seq_height 1080

# 2026.1.7

# windows version
python run.py --config E:/HWpro2026/GFVC/config/hdac.yaml

python HDAC_encoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/CFVQA_001_256x256_25_8bit_444.rgb"--encoding_frames 125 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 125

python HDAC_decoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/CFVQA_001_256x256_25_8bit_444.rgb"--encoding_frames 125 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 125

python HDAC_encoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/input1_001_256x256_394_8bit_444.rgb" --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

python HDAC_encoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/input1_001_256x256_394_8bit_420.yuv" --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

python HDAC_decoder.py  --original_seq "E:/HWpro2026/dataset/train/input1_001_256x256_394_8bit_420.yuv" --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

# linux version
python HDAC_encoder.py  --original_seq '/home/hanhan/datasets/gfvc/CFVQA_001_256x256_25_8bit_444.rgb' --encoding_frames 125 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 125

python run.py --config /home/hanhan/GFVC/config/hdac.yaml


