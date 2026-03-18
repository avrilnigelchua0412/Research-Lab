import albumentations as A
import pandas as pd
import os

class StaticVariable:
    main_raw_prefix = 'datasets/raw/'
    levels = pd.read_excel(f'{main_raw_prefix}fnab/FNAB ANNOTATIONS.xlsx', sheet_name='Classification')
    no_cluster_files = pd.read_excel(f'{main_raw_prefix}fnab/FNAB ANNOTATIONS.xlsx', sheet_name='Reannotations')
    summarized_levels = pd.read_csv('results/dataset_summary.csv') if os.path.exists('results/dataset_summary.csv') else None
    data_and_paths = pd.read_csv('results/explore_data_annotation_paths.csv') if os.path.exists('results/explore_data_annotation_paths.csv') else None
    formats = ['.jpeg', '.jpg', '.png']
    data_path = f'{main_raw_prefix}fnab'
    tile_size = 512
    min_pixel_size = 8
    value = [255, 255, 255]
    overlap = .25
    # {'confusant', 'CONFUSANT', 'Confusant', ' Confusant'}
    # {'Thycocyte', 'Cluster', 'Thyrocytes', 'Thyrocyte'}
    # label_map = {'Cluster' : 0, 'Clusters': 0, 'Thyrocyte': 1, 'Thyrocytes': 1, "Thycocyte": 1}
    # label_map = {'Cluster' : 0, 'Clusters': 0}
    label_map = {'Thyrocyte': 0, 'Thyrocytes': 0, "Thycocyte": 0, 'confusant' : 1, 'CONFUSANT' : 1, 'Confusant' : 1, ' Confusant' : 1}
    
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

    train_list = load_file_list('results/train_df_summary.csv')
    val_list   = load_file_list('results/val_df_summary.csv')
    test_list  = load_file_list('results/test_df_summary.csv')
    
    prefix = "datasets/processed/fnab/yolo_dataset_version_"
    
    tile_train_image_path = f"{prefix}2/images/train/"
    tile_train_label_path = f"{prefix}2/labels/train/"
    
    actual_train_image_path = f"{prefix}1/images/train/"
    actual_train_label_path = f"{prefix}1/labels/train/"
   
    tile_valid_image_path = f"{prefix}2/images/val/"
    tile_valid_label_path = f"{prefix}2/labels/val/"
    
    actual_valid_image_path = f"{prefix}1/images/val/"
    actual_valid_label_path = f"{prefix}1/labels/val/"
    
    tile_test_image_path = f"{prefix}2/images/test/"
    tile_test_label_path = f"{prefix}2/labels/test/"
    
    actual_test_image_path = f"{prefix}1/images/test/"
    actual_test_label_path = f"{prefix}1/labels/test/"
    
    DIR_PATH = [
        f"{prefix}1/images/train/",
        f"{prefix}1/images/val/",
        f"{prefix}1/images/test/",
        f"{prefix}1/labels/train/",
        f"{prefix}1/labels/val/",
        f"{prefix}1/labels/test/",
        f"{prefix}2/images/train/",
        f"{prefix}2/images/val/",
        f"{prefix}2/images/test/",
        f"{prefix}2/labels/train/",
        f"{prefix}2/labels/val/",
        f"{prefix}2/labels/test/"
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