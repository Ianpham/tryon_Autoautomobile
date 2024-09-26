import torch
import torch.nn as nn
import Image
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation
from transformers import SamModel, SamProcessor
import cv2
from PIL import image
class CropImage(nn.Module):
    def __init__(
        self,
        image_path,                 
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    ):
        super().__init__()
        self.car_label = car_label # 20 in this ade dataset
        self.feature_extractor = AutoFeatureExtractor(model_name)
        self.model = AutoModelForSemanticSegmentation(model_name)
        
        self.model.to(device)      
        
    def forward(self, img):
        if self.image_path:
           image = Image.open(self.image_path).convert("RGB")
        else:
            image = img

        # Perform car segmentation using SegFormer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.feature_extractor(image, return_tensors = "pt").to(device)
        image_np = np.array(image)
        with torch.no_grad():
            outputs = self.model(**inputs)
            car_mask = outputs.logits.argmax(dim = 1).squeeze().cpu().numpy()
        #convert the car segmentation mask into binary mask
        car_binary_mask = np.zeros_like(car_mask, dtype = np.int8)
        car_binary_mask[car_mask == 20] == 255 # class 20 represent to car class

        # resize car binary mask of the input image
        car_binary_mask = cv2.resize(car_binary_mask, (image_np.shape[1], image_np.shape[0]), interpolation = cv2.INTER_NEAREST)
        car_image = cv2.bitwise_and(image_np, image_np, mask = car_binary_mask)

        # find contours of binary mask
        contours, _ = cv2.findContours(car_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find the bounding box of the car
        if len(contours) >0:
            car_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(car_contour)

            # Crop the masked image to the bounding box
            cropped_image = car_image[y:y+h, x:x+w]

        else:
            cropped_image = car_image
        

        with torch.no_grad():
            result = self.model(image)

        predicted_mask = result.logits.argmax(dim = 1)[0].cpu().numpy()

        car_pixels = np.where(predicted_mask == self.car_label)

        if len(car_pixels) == 0:
            return ValueError("No car found in the picture")
        
        y1,x1 = np.min(car_pixels, dim = 1)
        y2,x2 = np.max(car_pixels, dim = 1)

        cropped_image = image.squeeze().permute(1,2,0).numpy()[y1:y2, x1:x2]

        return cropped_image

class Mask_generation(nn.Module):
    def __init__(self,
            model_name,
    ):
        super().__init__()
        self.processor = SamModel.from_pretrained(model_name)
        self.model = SamProcessor.from_pretrained(model_name)
        self.input_points = [[[450,600]]]     
    def forward(self,image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        device = "cuda" if torch.cuda.is_availabel() else "cpu"
        inputs = self.processor(image, input_points = self.input_points, return_tensors = "pt").to(device)
        outputs = self.model(**inputs)
        # Perform car segmentation using SegFormer
        with torch.no_grad():
            outputs = deep_model(**deep_inputs)
            mask = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),deep_inputs["original_sizes"].cpu(), deep_inputs["reshaped_input_sizes"].cpu())
        car_mask = np.zeros_like(mask, dtype = np.int8)
        for i, j in enumerate(np.unique(mask)):
            if np.isin(j, [0,1,2,3]):
                car_mask[mask == j] = 0
            else:
                car_mask[mask == j] = 255
        car_object = cv2.bitwise_and(image, image, mask = car_mask)

        # calculate the total area of the car
        total_car_area = np.sum(car_mask) // 255
        # set the threshold (exclude wheels and window)
        threshold = total_car_area // 7
        # Convert the car object to grayscale
        gray_car = cv2.cvtColor(car_object, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(gray_car, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on their size to identify the wheels and windows
        wheels_and_windows = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < threshold:
                wheels_and_windows.append(contour)

        # Create a mask for the wheels and windows
        wheels_windows_mask = np.zeros(car_object.shape[:2], dtype=np.uint8)
        cv2.drawContours(wheels_windows_mask, wheels_and_windows, -1, (255, 255, 255), -1)

        # Subtract the wheels and windows mask from the original car mask
        car_mask_without_wheels_windows = cv2.bitwise_and(car_mask, cv2.bitwise_not(wheels_windows_mask))

        # Apply the updated mask to the original image
        car_object_without_wheels_windows = cv2.bitwise_and(original_image, original_image, mask=car_mask_without_wheels_windows)

        return car_object_without_wheels_windows
