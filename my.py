import cv2
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# 参数配置
video_path = "./data/car-turn.mp4"    # 输入视频路径
output_dir = "./output/my-car-turn-opt-atten/"      # 输出目录
frame_interval = 5                # 帧间隔
prompt = "a red car turns in the winter"  # 替换为你的提示词
model_name = "runwayml/stable-diffusion-v1-5"  # 使用的模型

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化Stable Diffusion管道
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
).to(device)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("无法打开视频文件")

frame_count = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 按间隔提取帧
    if frame_count % frame_interval == 0:
        # 转换颜色空间 BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(frame_rgb)
        
        # 调整图像大小（根据模型需要调整）
        init_image = init_image.resize((512, 512))
        
        # 使用Stable Diffusion处理图像
        with torch.autocast(device):
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.5,       # 控制修改强度（0-1）
                guidance_scale=7.5, # 提示词相关性系数
                num_inference_steps=50
            ).images[0]
        
        # 保存结果
        output_path = os.path.join(output_dir, f"frame_{processed_count:04d}.jpg")
        result.save(output_path)
        processed_count += 1
        print(f"已处理 {processed_count} 帧")
    
    frame_count += 1

cap.release()
print("处理完成！")