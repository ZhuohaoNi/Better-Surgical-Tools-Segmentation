import cv2
import os
import numpy as np

def combine_domains_vertically():
    base_dir = "visualizations/Combined"
    output_path = os.path.join(base_dir, "all_domains_vertical.png")
    
    # Order: Bg Change, Regular, Low Brightness, Smoke, Blood
    domains_order = ['bg_change', 'regular', 'low_brightness', 'smoke', 'blood']
    
    images = []
    for domain in domains_order:
        # Construct path: visualizations/Combined/<domain>/combined_<domain>_0.png
        img_path = os.path.join(base_dir, domain, f"combined_{domain}_0.png")
        
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            return
            
        print(f"Loading {img_path}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image {img_path}")
            return
            
        images.append(img)
    
    # Check if all images have the same width
    widths = [img.shape[1] for img in images]
    if len(set(widths)) != 1:
        print(f"Warning: Images have different widths: {widths}. Resizing to match the first one.")
        target_width = widths[0]
        resized_images = []
        for img in images:
            if img.shape[1] != target_width:
                scale = target_width / img.shape[1]
                new_height = int(img.shape[0] * scale)
                resized_images.append(cv2.resize(img, (target_width, new_height)))
            else:
                resized_images.append(img)
        images = resized_images

    # Stack vertically
    print("Stacking images vertically...")
    final_image = np.concatenate(images, axis=0)
    
    cv2.imwrite(output_path, final_image)
    print(f"Saved combined image to {output_path}")

if __name__ == "__main__":
    combine_domains_vertically()
