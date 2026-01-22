# HDACencoder视频分辨率处理分析

## 1. 分辨率参数设置

HDACencoder通过命令行参数指定输入视频的分辨率：
- `--seq_width`：视频宽度，默认值为256
- `--seq_height`：视频高度，默认值为256

这些参数在代码中被解析并存储为变量：
```python
width = opt.seq_width
height = opt.seq_height
```

## 2. 原始视频读取

使用`raw_reader_planar`函数读取原始视频序列，传入指定的宽度和高度参数：
```python
listR, listG, listB = raw_reader_planar(original_seq, width, height, frames)
```

## 3. 分辨率调整函数

代码中定义了`resize_frames`函数用于调整视频帧的分辨率：
```python
def resize_frames(frames: np.ndarray, scale_factor=1)->np.ndarray:
    N, H, W, C = frames.shape
    if scale_factor != 1:
        out = []
        for idx in range(N):
            img = Image.fromarray(frames[idx])
            img = img.resize((int(H*scale_factor), int(W*scale_factor)), resample=Image.Resampling.LANCZOS)
            out.append(np.asarray(img))
        return np.array(out)
    else:
        return frames
```

该函数：
- 使用PIL库的Image.resize方法进行缩放
- 采用LANCZOS重采样算法保证图像质量
- 当scale_factor为1时不进行缩放，直接返回原帧

## 4. 基础层分辨率处理

当启用基础层编码（use_base_layer == 'ON'）时，会对基础层进行分辨率处理：

### 4.1 基础层编码前缩放
```python
bl_coding_params = {
    'qp': bl_qp,
    'sequence': resize_frames(gop, bl_scale_factor),
    'fps': 25,
    'frame_dim': (h, w),
    'log_path': bl_output_path
}
```

### 4.2 基础层解码后恢复
```python
bl_frames = np.transpose(resize_frames(info_out['dec_frames'], 1//bl_scale_factor), [0, 3, 1, 2])
```

基础层缩放因子通过`--bl_scale_factor`参数指定，默认值为1.0（不缩放）。

## 5. 中间帧分辨率处理

在处理中间帧时，确保其分辨率与指定的输入分辨率一致：
```python
inter_frame = cv2.merge(current_frame)
inter_frame = resize(inter_frame, (width, height))[..., :3]
```

这里使用`resize`函数（推测来自cv2库）将中间帧调整到指定的width和height。

## 6. 分辨率处理流程总结

1. **输入指定**：通过命令行参数指定目标分辨率
2. **原始读取**：使用指定分辨率读取原始视频
3. **基础层处理**：
   - 编码前：按缩放因子缩小基础层
   - 解码后：恢复到原始分辨率
4. **中间帧处理**：确保中间帧分辨率与目标分辨率一致
5. **参考帧处理**：直接使用指定分辨率，不进行额外缩放

## 7. 关键参数

- `--seq_width`/`--seq_height`：目标编码分辨率
- `--bl_scale_factor`：基础层缩放因子

## 8. 注意事项

1. 分辨率调整使用高质量的LANCZOS重采样算法
2. 基础层缩放是可选的，默认不缩放
3. 所有中间帧都会被调整到指定的目标分辨率
4. 输入视频的实际分辨率应与指定的参数匹配，否则可能导致读取错误或图像变形