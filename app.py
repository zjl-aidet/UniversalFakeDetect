
import torch
import torch.nn
import torchvision.transforms as transforms
import gradio as gr
from models import get_model


# 加载与训练中使用的相同结构的模型
def load_model(arch, checkpoint_path):
    model = get_model(arch) 
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    # model.cuda()
    model.eval()
    return model


# 加载图像并执行必要的转换的函数
def process_image(image, image_size):
    trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] ),
    ])

    image = trans(image.convert('RGB'))
    return image

# 预测图像类别并返回概率的函数
def predict(image):
    classes = {'0': 'fake', '1': 'real'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        in_tens = image.unsqueeze(0)
        # in_tens = in_tens.cuda()
        prob = model(in_tens).sigmoid().item()
        print(prob)
        probabilities = (float(prob), 1-(prob))
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}

    return class_probabilities


# 定义到您的模型权重的路径
arch='CLIP:ViT-L/14'
checkpoint_path = './checkpoints/clip_vitl14/model_epoch_best.pth'
model = load_model(arch, checkpoint_path)
num_classes = 2

# 定义Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Fake vs Real Classifier",
    examples=["examples/fake.png", "examples/real.png"]
)

if __name__ == "__main__":
    iface.launch(share=True, server_name='0.0.0.0', server_port=443)
