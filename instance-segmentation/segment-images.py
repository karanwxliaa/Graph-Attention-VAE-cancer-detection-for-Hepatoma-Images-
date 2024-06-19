import numpy as np
from PIL import Image
import os

from csbdeep.utils import normalize
from stardist.models import StarDist2D


# Function to load image and convert to numpy array
def load_image(img_path):
    img = Image.open(img_path).resize((1600, 1600))
    img = np.array(img)
    return normalize(img, 1, 99.8)


def segment_and_save_images(model, root_dir, save_dir, normalizer=None):
    for category in os.listdir(root_dir):
        print(f"Currently segmenting images from category: {category} !")
        os.makedirs(os.path.join(save_dir, category), exist_ok=True)
        category_path = os.path.join(root_dir, category)
        img_paths = []
        img_names = []
        for img in os.listdir(category_path):
            img_paths.append(os.path.join(category_path, img))
            img_names.append(img)
        images = [load_image(img_path) for img_path in img_paths]
        idx = 0
        for img in images:
            idx += 1
            try:
                labels, _ = model.predict_instances(img, prob_thresh=0.20)
                labels[labels > 0] = 1
                labels = (labels * 255).astype('uint8')
                # Convert the numpy array to an image
                image = Image.fromarray(labels)
                # Save the image
                image.save(os.path.join(save_dir, category, f"{img_names[idx]}.png"))
            except:
                print(f"Image at index {idx} failed to be segmented due to exception !")
                pass

        print(f"Completed segmenting images from category: {category} !\n\n")
        

if __name__ == '__main__':

    model = StarDist2D.from_pretrained('2D_versatile_he')

    root_dir = r"D:\\VIT Material\\VIT material\\Hepatoma Research Project\\Histopathology-Images"
    save_dir = r"D:\\VIT Material\\VIT material\\Hepatoma Research Project\Segmented-Images"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting Segmentation...\n\n")
    segment_and_save_images(model, root_dir, save_dir)

    print(f"Segmented and saved images to save directory: {save_dir} !")
    