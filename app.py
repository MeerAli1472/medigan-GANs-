# Import needed libraries
import gradio as gr
import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# Define the GAN models available in the app
model_ids = [
    "00001_DCGAN_MMG_CALC_ROI",
    "00002_DCGAN_MMG_MASS_ROI",
    "00003_CYCLEGAN_MMG_DENSITY_FULL",
    "00004_PIX2PIX_MMG_MASSES_W_MASKS",
    "00019_PGGAN_CHEST_XRAY"
]

def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=True,
        num_samples=num_images,
        prefetch_factor=None,
    )

    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4:
                    sample = sample.squeeze(0).permute(2, 0, 1)

                sample = to_pil_image(sample).convert("RGB")
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                ])
                sample = transform(sample)
                image_list.append(sample)

            if "mask" in i:
                mask = data_dict.get("mask")
                if mask.dim() == 4:
                    mask = mask.squeeze(0).permute(2, 0, 1)
                mask = to_pil_image(mask).convert("RGB")
                mask = transform(mask)
                image_list.append(mask)

        grid = make_grid(image_list, nrow=2)

        if grid.dim() == 4:
            grid = grid.squeeze(0)
            if grid.size(-1) == 1:
                grid = grid.squeeze(-1)
            else:
                raise ValueError("Expected a single channel (grayscale) image.")

        img = torchvision.transforms.ToPILImage()(grid)
        images.append(img)
    return images


def generate_images_gradio(model_id, num_images):
    images = torch_images(num_images, model_id)
    return images  # Return list of images for Gradio's Image gallery

# Define the Gradio UI components
iface = gr.Interface(
    fn=generate_images_gradio,
    inputs=[
        gr.Dropdown(choices=model_ids, label="Select Model ID"),
        gr.Slider(minimum=1, maximum=7, step=1, label="Number of Images", value=1),
    ],
    outputs=gr.Gallery(
    label="Generated Images",
    columns=2,
    height="auto"),

    title="MEDIGAN Medical Image Data Generator",
    description="Select a GAN model and generate synthetic medical images."
)

# Launch the app
iface.launch(share=True)

