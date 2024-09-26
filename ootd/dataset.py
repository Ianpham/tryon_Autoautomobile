import os
import random
import math
import numpy as np
import psutil
import shutil
import time
import json
from multiprocessing.pool import ThreadPool
import tqdm

import requests
from io import BytesIO
import instaloader
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.by import By
from pathlib import Path

import cv2
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import LOGGER


class CarTryOnDataset(Dataset):
    def __init__(
            self,
            stanford_cars_root_dir,
            instagram_hashtag,
            instagram_num_posts,
            unsplash_access_key,
            unsplash_query_car,
            unsplash_query_decal,
            unsplash_per_page,
            pexels_api_key,
            pexels_search_query,
            pexels_num_images,
            redbubble_search_query,
            redbubble_num_images,
            google_image_scraper_car_keyword,
            google_image_scraper_decal_keyword,
            google_image_scraper_limit,
            hyp=None,
            transform=None,
            cache_ram = None):
        super().__init__()
        self,detection_model = self._load_yolov8_model()
        self.stanford_cars_dataset = StanfordCarsDataset(root_dir=stanford_cars_root_dir, transform=transform)
        self.instagram_dataset = InstagramDataset(hashtag=instagram_hashtag, num_posts=instagram_num_posts, transform=transform)
        self.unsplash_dataset_car = UnsplashDataset(access_key=unsplash_access_key, query=unsplash_query_car, per_page=unsplash_per_page)
        self.unsplash_dataset_decal = UnsplashDataset(access_key=unsplash_access_key, query=unsplash_query_decal, per_page=unsplash_per_page)
        self.pexels_dataset = PexelsDataset(api_key=pexels_api_key, search_query=pexels_search_query, num_images=pexels_num_images, transform=transform)
        self.redbubble_dataset = RedbubbleDataset(search_query=redbubble_search_query, num_images=redbubble_num_images, transform=transform)
        self.google_image_scraper_dataset_car = GoogleImageScraperDataset(keyword=google_image_scraper_car_keyword, limit=google_image_scraper_limit)
        self.google_image_scraper_dataset_decal = GoogleImageScraperDataset(keyword=google_image_scraper_decal_keyword, limit=google_image_scraper_limit)
        self.hyp = hyp
        self.transform = transform if transform is not None else self.mosaic_augmentation()
        self.cache_ram = cache_ram
        # this is our target image package look like
        # prompt = batch['prompt']
        # image_garm = batch['image_garm']
        # image_vton = batch['image_vton']
        # mask = batch['mask']
        # image_ori = batch['image_ori'] #consider wherether it need or not
        # target_image = batch['target_image']

        # Process and gather data from other datasets
        self.car_images = []
        self.car_masks = []
        self.design_images = []

        # Stanford Cars Dataset
        for image, label in self.stanford_cars_dataset:
            self.car_images.append(image)
            mask = self._generate_mask(image)
            self.car_masks.append(mask)

        # Instagram Dataset
        for image in self.instagram_dataset:
            self.car_images.append(image)
            mask = self._generate_mask(image)
            self.car_masks.append(mask)

        # Unsplash Dataset
        for image in self.unsplash_dataset_car:
            self.car_images.append(image)
            mask = self._generate_mask(image)
            self.car_masks.append(mask)
        for image in self.unsplash_dataset_decal:
            self.design_images.append(image)

        # Pexels Dataset
        for image in self.pexels_dataset:
            self.car_images.append(image)
            mask = self._generate_mask(image)
            self.car_masks.append(mask)

        # Redbubble Dataset
        for image in self.redbubble_dataset:
            self.design_images.append(image)

        # Google Image Scraper Dataset
        for image_data in self.google_image_scraper_dataset_car:
            image_path = image_data["link"]
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            self.car_images.append(image)
            mask = self._generate_mask(image)
            self.car_masks.append(mask)
        for image_data in self.google_image_scraper_dataset_decal:
            image_path = image_data["link"]
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            self.design_images.append(image)

        if self.cache_ram:
            self.num_imgs = len(self.car_images) + len(self.car_masks) + len(self.design_images)
            self.imgs =  [None] * self.num_imgs
            self.cache_image(self.car_images, num_imgs = self.num_imgs)
    # for creating mask on the image, please consider with masked car(exclude decal not background), so we are able to do mask on the image
    def crop_image_to_car(self,image):
        results = self.detection_model(image)
        car_boxes = []

        for result in results:
            for detection in result.boxes.data.to_list():
                class_id = int(detection[5])
                if class_id == 2:
                    car_boxes.append(detection[:4])

        if len(car_boxes) > 0:
            left, top, right, bottom = map(int, car_boxes[0])
            car_image = image.crop((left, top, right, bottom))
        else:
            # If no car instances are detected, use the original image
            car_image = image

        return car_image
    def _generate_mask_(self, image):
        results = self.detection_model(image)
        car_masks = []
        car_boxes = []

        for result in results:
            for detection in result.boxes.data.to_list():
                class_id = int(detection[5])
                if class_id == 2:
                    mask = result.masks.data[result.boxes.cls == class_id].sequeeze.cpu().numpy()
                    car_boxes.append(detection[:4])

        if len(car_masks) > 0:
            combined_mask = Image.new("L", image.size, 0)
            for mask in car_masks:
                combined_mask = Image.fromarray(np.maximum(combined_mask, ,mask))

            # crop the car image based on the bounding box
            car_boxe = car_boxes[0]
            left, top, right, bottom = map(int(car_boxe))
            car_image = image.crop((left, top, right, bottom))
        else:
            # If no car instances are detected, use the original image and create an empty mask
            combined_mask = Image.new("L", image.size, 0)
            car_image = image

        return car_image, combined_mask

    def _load_yolov8_model(self):
        model = YOLO("kermemberke/yolov8m-seg")
        model.fuse()
        return model
    
    # for exectution, wwe need object detection for delete what is unnecessary
    def __len__(self):
        return len(self.car_images)

    def __getitem__(self, index):
        image = self.car_images[index]
        car_image, car_mask = self._generate_mask_and_crop(image)
        design_image = self.design_images[index % len(self.design_images)]

        if self.transform:
            car_image = self.transform(car_image)
            car_mask = self.transform(car_mask)
            design_image = self.transform(design_image)

        return {
            "car_image": car_image,
            "car_mask": car_mask,
            "design_image": design_image
        }

    
    def _get_file_list(self, directory):
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def random_affine(self, image, labels = (), degrees = 10, translate = .1, scale = .1, shear = 10, new_shape = (640,640)):
        ''' Applied affine transformatuin'''

        # for later labels n = len(labels)
        if isinstance(new_shape, int):
            height = width = new_shape
        else:
            height,width = new_shape

        M,s = self.get_transform_matrix(image, new_shape, degrees, scale, shear, translate)

        if (M != np.eyes(3)).any():
            img = cv2.warpAffine(img, M[:2], dsize = (width, height), borderValue = (114,114,114))

        return img
    
    # task: define hs, ws, hyp for this function along with class
    def mosaic_augmentation(self,shape, image,hs, ws, hyp,specific_shape = False, target_height = 640, target_width = 640):
        ''' applied mosaic augmentation'''

        assert len(image) == 4,  "Mosaic augmentation of current version only supports 4 images."
        if not specific_shape:
            if isinstance(shape, list) or isinstance(shape, np.ndarray):
                target_height, target_width = shape

            else:
                target_height, target_width = shape

        yc, xc = (int(random.uniform(x//2, 3*x//2)) for x in (target_height, target_width))

        for i in range(len(image)):
            img, h, w = image[i], hs[i], ws[i]
            if i == 0:
                img4 = np.full((target_height *2, target_width *2, img.shape[2]))
                 # place img in img4
            if i == 0:  # top left
                img4 = np.full((target_height * 2, target_width * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        img4 = self.random_affine(img4,
                            degrees=hyp['degrees'],
                            translate=hyp['translate'],
                            scale=hyp['scale'],
                            shear=hyp['shear'],
                            new_shape=(target_height, target_width))
        
        return img4

    def get_transform_matrix(image_shape, new_shape, degrees, scale, shear, translate):
        new_height, new_width = new_shape
        #center
        C = np.eye(3)
        C[0,2] = -image_shape[1]/2
        C[1,2] = image_shape[0]/2

        # rotate and scale
        R = np.eye(3)
        d =  random.uniform(-degrees, degrees)

        s = random.uniform(1-scale, 1 +scale)

        R[:2] = cv2.getRotationMatrix2D(angle =d, center = (0,0), scale = s)

        # shear
        S = np.eye(3)
        S[0,1] = math.tan(random.uniform(-shear, shear)*math.pi/180) #(x)
        S[1,0] = math.tan(random.uniform(-shear, shear)*math.pi/180) #(y)

        # translation

        T = np.eye(3)
        T[0,2] = random.uniform(0.5 - translate, 0.5 + translate)*new_width
        T[1,2] = random.uniform(0.5 - translate, 0.5 + translate)*new_height

        M = T @ S @ R @ C

        return M, s
    # copied from https://github.com/meituan/YOLOv6/blob/main/yolov6/data/datasets.py#L114
    def cache_image(self, image_path,num_imgs = None,):
        # image_path shall be self.image_path, but we have two source of images, that mean calculation shall be done twice.
        assert num_imgs is not None, "num_imgs must be specified ass the size of dataset"
        mem = psutil.virtual_memory()
        mem_required = self.cal_cache_occupy(image_path,num_imgs)
        gb = 1 << 30
        if mem_required > mem.available:
            self.cache_ram = False
            LOGGER.warning("Not enough Ram to cache images, caching is disabled")
        else:
            LOGGER.warning(
                f"{mem_required / gb:.1f} GB RAM required, "
                f"{mem.available / gb:.1f}/{mem.total/ gb:.1f} GB RAM available, "
                f"Since the first thing we do is cache, "
                f"there is no guarantee that the remaining memory space is sufficient"
            )

        print(f"self.imgs: {len(self.imgs)}")
        LOGGER.info("you are using cached images in RAM to accelerate training")
        LOGGER.info(
            "Caching images ...\n"
            "This might take some time for your dataset"
        )
        num_threads = min(16, max(1, os.cpu_count() - 1))
        load_imgs = ThreadPool(num_threads).imap(self.load_image, range(num_imgs))
        pbar = tqdm(enumerate(load_imgs), total=num_imgs, disable=self.rank > 0)
        for i, (x, (h0, w0), (h1,w1)) in pbar:
            self.imgs[i], self.imgs_hw0[i], self.imgs_hw[i] = x, (h0, w0), (h1, w1)
    def cal_cache_occupy(self, image_path, num_imgs):
        """estimate memory required to cache images in Ram"""
        cache_bytes = 0
        num_imgs = len(image_path)
        num_samples = min(num_imgs, 32)
        for _ in range(num_samples):
            img,_,_ = self.load_image(index = random.randint(0, len(image_path)-1))
            cache_bytes += img.nbytes
        mem_required = cache_bytes * num_imgs / num_samples
        return mem_required
    
    def load_image(self, index, target_size=None):
        image = self.car_images[index]
        
        if target_size is not None:
            # Resize the image to the target size
            image = image.resize(target_size)
        
        # Convert the image to numpy array
        image_np = np.array(image)
        
        # Get the original image size
        h0, w0 = image_np.shape[:2]
        
        # Calculate the resize ratio based on the target size
        if target_size is not None:
            ratio = min(target_size[0] / w0, target_size[1] / h0)
        else:
            ratio = 1.0
        
        # Resize the image while maintaining the aspect ratio
        if ratio != 1.0:
            interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
            image_np = cv2.resize(image_np, None, fx=ratio, fy=ratio, interpolation=interpolation)
        
        # Get the new image size
        h1, w1 = image_np.shape[:2]
        
        return image_np, (h0, w0), (h1, w1)

# Create the transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the dataset instance
dataset = CarTryOnDataset(
    car_dataset_dir="path/to/stanford_cars_dataset", #(8126, 64, 64, 3)=> please consider how small it is.
    design_dataset_dir="path/to/vehicle_semantic_paint_dataset",
    transform=transform
)

# Create the dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

class UnsplashDataset(Dataset):
    def __init__(self,
                 access_key,
                 querry, # querry shalle be "car" or "decal" in this situation
                 per_page = 30,
    ):
        self.access_key = access_key
        self.querry = querry 
        self.per_page = per_page
        self.data = self.fetch_data()

    def fetch_data(self):
        url = f"https://api.unsplash.com/search/photos?query={self.querry}&per_page={self.per_page}&client_id={self.access_key}"
        header = {"Authorization": f"Client-ID {self.access_key}"}
        response = requests.get(url, headers= header)
        data = response.json()
        return data["results"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        image_url = item["urls"]["regular"]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        return image


class StanfordCarsDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split,
            transform = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(self.root_dir, self.split, "car_ims")
        self.anno_dir = os.path.join(self.root_dir, self.split, "anno_ims")

        self.image_files = []
        self.labels = []

        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file,'r') as f:
            for line in f:
                image_file, label = line.strip().split()
                self.image_files.append(image_file)
                self.labels.append(label)

    def __len__(self):
        return(len(self.image_files))
    
    def __getitem__(self, index):
        image_file = self.image_files(index)
        label = self.labels[index]

        image_path = os.path.join(self.img_dir, image_file)
        image  = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


class InstagramDataset(Dataset):
    def __init__(
            self,
            hashtag,
            num_posts,
            transform = None
    ):
        self.hashtag = hashtag
        self.num_posts = num_posts
        self.transform = transform

        self.loader = instaloader.Instaloader()
        self.posts = []

        for post in instaloader.Hashtag.from_name(self.loader.context, self.hashtag).get_posts():
            self.posts.append(post)
            if len(self.posts) == self.num_posts:
                break

    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, index):

        posts = self.posts[index]
        image_url = posts.url

        # download image
        self.loader.download_pic(image_url, target = f"{self.hashtag}_{index}.png")
        image_path = f"{self.hashtag}_{index}.jpg"

        # load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # remove downloaded image file
        os.remove(image_path)
        
        return image
    

    def fetch_data(self):
        pass
    

class PexelsDataset(Dataset):
    def __init__(self, api_key, search_query, num_images, transform=None):
        self.api_key = api_key
        self.search_query = search_query
        self.num_images = num_images
        self.transform = transform
        self.images = []
        
        # Make API request to Pexels
        url = f"https://api.pexels.com/v1/search?query={search_query}&per_page={num_images}"
        headers = {"Authorization": self.api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            self.images = [photo["src"]["large"] for photo in data["photos"]]
        else:
            print(f"Failed to fetch images from Pexels. Status code: {response.status_code}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_url = self.images[index]
        
        # Download image
        response = requests.get(image_url)
        with open(f"image_{index}.jpg", "wb") as file:
            file.write(response.content)
        
        # Load and transform image
        image_path = f"image_{index}.jpg"
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        # Remove downloaded image file
        os.remove(image_path)
        
        return image
    
class RedbubbleDataset(Dataset):
    def __init__(self, search_query, num_images, transform=None):
        self.search_query = search_query
        self.num_images = num_images
        self.transform = transform
        self.images = []
        
        # Scrape Redbubble search results
        url = f"https://www.redbubble.com/shop/?query={search_query}"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            image_elements = soup.select("img.styles__productImage--1wZzz")
            
            for img in image_elements[:num_images]:
                image_url = img["src"]
                self.images.append(image_url)
        else:
            print(f"Failed to fetch images from Redbubble. Status code: {response.status_code}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_url = self.images[index]
        
        # Download image
        response = requests.get(image_url)
        with open(f"image_{index}.jpg", "wb") as file:
            file.write(response.content)
        
        # Load and transform image
        image_path = f"image_{index}.jpg"
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        # Remove downloaded image file
        os.remove(image_path)
        
        return image
    
class GoogleImageScraperDataset(Dataset):
    def __init__(self, keyword, metadata=False, limit=1000):
        self.keyword = keyword
        self.metadata = metadata
        self.limit = limit
        self.image_data = []
        self.scrape_images()

    def scrape_images(self):
        url = f"https://www.google.com/search?q={self.keyword}&source=lnms&tbm=isch"
        source = self.search(url)
        soup = BeautifulSoup(str(source), "html.parser")
        
        links = [json.loads(i.text)["ou"] for i in soup.find_all("div", class_="rg_meta")]
        print(f"[%] Indexed {len(links)} Possible Images.")
        print("\n===============================================\n")
        print("[%] Getting Image Information.")

        for a in soup.find_all("div", class_="rg_meta"):
            if len(self.image_data) >= self.limit:
                break

            rg_meta = json.loads(a.text)
            title = rg_meta.get('st', '')
            link = rg_meta["ou"]

            try:
                image_data = {
                    "source": "google",
                    "keyword": self.keyword,
                    "title": rg_meta["pt"],
                    "size": rg_meta["s"],
                    "description": title,
                    "link": link,
                    "origin": rg_meta["ru"]
                }
                self.image_data.append(image_data)
                self.download_image(link, image_data)
            except Exception as e:
                print(f"[!] Issue getting data: {rg_meta}\n[!] Error: {e}")

        print(f"\n\n[%] Done. Downloaded {len(self.image_data)} images.")
        print("\n===============================================\n")

    def search(self, url):
        firefox_options = Options()
        firefox_options.add_argument("--headless")
        firefox_profile = webdriver.FirefoxProfile()
        firefox_options.profile = firefox_profile
        browser = webdriver.Firefox(options=firefox_options)
        browser.implicitly_wait(30)
        browser.set_window_size(1024, 768)
        browser.get(url)
        time.sleep(1)

        element = browser.find_element(By.TAG_NAME, "body")
        for _ in range(30):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

        try:
            browser.find_element(By.ID, "smb").click()
            for _ in range(50):
                element.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
        except Exception:
            for _ in range(10):
                element.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)

        source = browser.page_source
        browser.close()
        
        return source

    def download_image(self, link, image_data):
        try:
            file_name = link.split("/")[-1]
            file_type = file_name.split(".")[-1]
            file_type = file_type[:3] if len(file_type) > 3 else file_type
            if file_type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
                file_type = "jpg"

            save_path = f"{os.getcwd()}/dataset/google/{self.keyword}/Scrapper_{len(self.image_data)}.{file_type}"
            self.save_image(link, save_path)
            print(f"[%] Downloaded File: {save_path}")

            if self.metadata:
                metadata_path = f"{os.getcwd()}/dataset/google/{self.keyword}/Scrapper_{len(self.image_data)}.json"
                with open(metadata_path, "w") as outfile:
                    json.dump(image_data, outfile, indent=4)
        except Exception as e:
            print(f"[!] Issue Downloading: {link}\n[!] Error: {e}")
            self.error(link)

    def save_image(self, link, file_path):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(link, stream=True, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, file)
        else:
            raise Exception(f"Image returned a {response.status_code} error.")

    def error(self, link):
        print(f"[!] Skipping {link}. Can't download or no metadata.")
        error_file = Path(f"{os.getcwd()}/dataset/logs/google/errors.log")
        with open(error_file, "a") as file:
            file.write(link + "\n")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.image_data[idx]