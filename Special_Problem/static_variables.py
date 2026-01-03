import albumentations as A
import pandas as pd
import os

class StaticVariable:
    levels = pd.read_excel('/workspace/Special_Problem/FNAB ANNOTATIONS.xlsx', sheet_name='Classification')
    no_cluster_files = pd.read_excel('/workspace/Special_Problem/FNAB ANNOTATIONS.xlsx', sheet_name='Reannotations')
    summarized_levels = pd.read_csv('/workspace/Special_Problem/dataset_summary.csv') if os.path.exists('/workspace/Special_Problem/dataset_summary.csv') else None
    data_and_paths = pd.read_csv('/workspace/Special_Problem/explore_data_annotation_paths.csv') if os.path.exists('/workspace/Special_Problem/explore_data_annotation_paths.csv') else None
    formats = ['.jpeg', '.jpg', '.png']
    data_path = '/workspace/Special_Problem/Data'
    tile_size = 512
    min_pixel_size = 8
    value = [255, 255, 255]
    overlap = .25
    # {'Thycocyte', 'Cluster', 'Thyrocytes', 'Thyrocyte'}
    # label_map = {'Cluster' : 0, 'Clusters': 0, 'Thyrocyte': 1, 'Thyrocytes': 1, "Thycocyte": 1}
    # label_map = {'Cluster' : 0, 'Clusters': 0}
    label_map = {'Thyrocyte': 0, 'Thyrocytes': 0, "Thycocyte": 0}
    
    def load_file_list(path, return_none=True):
        """
        Reads a CSV file and returns the 'File' column as a list.
        If the file does not exist, returns None or [] depending on `return_none`.
        """
        if not os.path.exists(path):
            return None if return_none else []

        try:
            df = pd.read_csv(path)
            return df['File'].to_list() if 'File' in df.columns else None
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None if return_none else []

    train_list = load_file_list('/workspace/Special_Problem/train_df_summary.csv')
    val_list   = load_file_list('/workspace/Special_Problem/val_df_summary.csv')
    test_list  = load_file_list('/workspace/Special_Problem/test_df_summary.csv')
    
    tile_train_image_path = "/workspace/Special_Problem/yolo_dataset_version_2/images/train/"
    tile_train_label_path = "/workspace/Special_Problem/yolo_dataset_version_2/labels/train/"
    
    actual_train_image_path = "/workspace/Special_Problem/yolo_dataset_version_1/images/train/"
    actual_train_label_path = "/workspace/Special_Problem/yolo_dataset_version_1/labels/train/"
   
    tile_valid_image_path = "/workspace/Special_Problem/yolo_dataset_version_2/images/val/"
    tile_valid_label_path = "/workspace/Special_Problem/yolo_dataset_version_2/labels/val/"
    
    actual_valid_image_path = "/workspace/Special_Problem/yolo_dataset_version_1/images/val/"
    actual_valid_label_path = "/workspace/Special_Problem/yolo_dataset_version_1/labels/val/"
    
    tile_test_image_path = "/workspace/Special_Problem/yolo_dataset_version_2/images/test/"
    tile_test_label_path = "/workspace/Special_Problem/yolo_dataset_version_2/labels/test/"
    
    actual_test_image_path = "/workspace/Special_Problem/yolo_dataset_version_1/images/test/"
    actual_test_label_path = "/workspace/Special_Problem/yolo_dataset_version_1/labels/test/"
    
    # tile_path = '/workspace/Special_Problem/yolo_dataset_version_2/'

    DIR_PATH = [
        "/workspace/Special_Problem/yolo_dataset_version_1/images/train/",
        "/workspace/Special_Problem/yolo_dataset_version_1/images/val/",
        "/workspace/Special_Problem/yolo_dataset_version_1/images/test/",
        "/workspace/Special_Problem/yolo_dataset_version_1/labels/train/",
        "/workspace/Special_Problem/yolo_dataset_version_1/labels/val/",
        "/workspace/Special_Problem/yolo_dataset_version_1/labels/test/",
        "/workspace/Special_Problem/yolo_dataset_version_2/images/train/",
        "/workspace/Special_Problem/yolo_dataset_version_2/images/val/",
        "/workspace/Special_Problem/yolo_dataset_version_2/images/test/",
        "/workspace/Special_Problem/yolo_dataset_version_2/labels/train/",
        "/workspace/Special_Problem/yolo_dataset_version_2/labels/val/",
        "/workspace/Special_Problem/yolo_dataset_version_2/labels/test/"
        ]
        
    transform_level_1 = A.Compose(
        [
            # Geometric Transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.95, 1.05),
                rotate=(-15, 15),
                p=.75),
            
            # Photometric Transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.05), p=0.5), # 3% to 5% noise
            
            # Occlusion/regularization
            A.CoarseDropout(
            num_holes_range=(5, 5),
            hole_height_range=(20, 20),
            hole_width_range=(20, 20),
            fill="random_uniform",
            p=0.5),
            A.GridDropout(ratio=0.05, p=0.5)
        ],
        seed=42,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'],)
    )
    
    transform_level_2 = A.Compose(
        [
            # Geometric Transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=0.03,
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                p=0.5),
            
            # Photometric Transformations
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.GaussNoise(std_range=(0.03, 0.05), p=0.3), # 3% to 5% noise
            
        ],
        seed=42,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'],)
    )
    
    transform_level_3_to_4 = A.Compose(
        [
            # Geometric Transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=0.02,
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                p=0.3),
            
            # Photometric Transformations
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
        ],
        seed=42,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'],)
    )
    
    transform_level_5 = A.Compose([
            # Geometric Transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        seed=42,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'],)
    )
    
    @classmethod
    def is_supported(cls, ext):
        return ext.lower() in cls.formats
    
    @classmethod
    def get_transform(cls):
        return cls.transform
    
    # @staticmethod
    # def cluster_group(x):
    #     if x <= 4:
    #         return "low"
    #     elif x <= 10:
    #         return "medium"
    #     else:
    #         return "high"