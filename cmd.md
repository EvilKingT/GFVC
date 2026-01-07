


# 2026.1.7
python run.py --config E:/HWpro2026/GFVC/config/hdac.yaml

python HDAC_encoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/CFVQA_001_256x256_25_8bit_444.rgb"--encoding_frames 125 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 125

python HDAC_decoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/CFVQA_001_256x256_25_8bit_444.rgb"--encoding_frames 125 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 125

python HDAC_encoder_v1.py  --original_seq "E:/HWpro2026/dataset/train/input1_001_256x256_394_8bit_420.yuv" --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394

python HDAC_decoder.py  --original_seq "E:/HWpro2026/dataset/train/input1_001_256x256_394_8bit_420.yuv" --encoding_frames 394 --quantization_factor 64 --iframe_qp 32 --iframe_format YUV420 --ref_codec vtm --adaptive_metric PSNR --adaptive_thresh 0 --num_kp 10 --use_base_layer ON --base_codec hevc --bl_qp 51 --bl_scale_factor 1  --gop_size 394