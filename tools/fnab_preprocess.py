import os
from static_variables import StaticVariable
import re
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image, ImageDraw
import warnings

ERROR = set()

class Utils:
    @staticmethod
    def check_dataset():
        rows = []
        for image_path, _ in Utils.helper_os_walk():
            a_path, b_path  = Utils.replace(image_path)
            if not os.path.exists(a_path) or not os.path.exists(b_path):
                rows.append(image_path)
        return rows
    
    @staticmethod
    def replace(original_string, affixes=[' - ANNOTATED FILES', 'A.csv', 'B.json']):
        match_batch_num = re.search(r'BATCH (\d+)', original_string)
        match_format = re.search(r'\.(jpeg|jpg|png)$', original_string)
        # Guard clause
        if match_batch_num is None or match_format is None:
            return "", ""
        replace = re.sub(r'BATCH \d+', f'BATCH {match_batch_num.group(1)}{affixes[0]}', original_string)
        a_path = re.sub(match_format.group(0), affixes[1], replace)
        b_path = re.sub(match_format.group(0), affixes[2], replace) if len(affixes) > 2 else None
        return a_path, b_path if len(affixes) > 2 else None
    
    @staticmethod
    def updated_replace(path, file_name):
        path = re.sub(r'BATCH \d+', f'Thyrocytes and Clusters - Update', path)
        candidates = [
            f"{file_name}-thyrocyte and cluster.csv",
            f"{file_name}-thyrocytes.csv"
        ]
        directory = os.path.dirname(path)
        for candidate in candidates:
            if os.path.exists(os.path.join(directory, candidate)):
                return re.sub(
                    rf'{re.escape(file_name)}\.(jpeg|jpg|png)$',
                    candidate,
                    path
                )
        return None
    
    @staticmethod
    def get_json_data(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
        
    @staticmethod
    def get_csv_data(csv_path, file):
        df = pd.read_csv(csv_path)

        # Identify trash BEFORE filtering
        trash_data = df[df["image_name"] != file]

        # Identify correct rows
        valid_data = df[df["image_name"] == file]

        if not trash_data.empty:
            # print(f"[WARNING] Found trash rows (image_name != {file}):")
            # print(file.split('.')[0].split('-')[-1])
            ERROR.add(file.split('.')[0].split('-')[-1])
            # print(trash_data)

        return valid_data
    
    @staticmethod
    def get_image_data(image_path):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    @staticmethod
    def visualize_bboxes(bboxes, labels, ax):
        """Draw bounding boxes with color-coded labels on a Matplotlib axis."""
        for box, label in zip(bboxes, labels):
            x_min, y_min, box_width, box_height = box
            # choose color based on label
            color = "red" if label == 'Cluster' or label == 'Clusters' else "black"
            # choose linewidth
            linewidth = 1 if label == 'Cluster' or label == 'Clusters' else 1.5
            label = "Cluster" if color == 'red' else "Thyrocyte"
            # draw bounding box
            ax.add_patch(plt.Rectangle(
                (x_min, y_min),
                box_width, box_height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none"
            ))
            # draw label text
            ax.text(
                x_min, y_min - 5,
                label,
                color=color,
                fontsize=8,
                weight="bold"
            )
    
    @staticmethod
    def image_tiling(original_image, tile_size=StaticVariable.tile_size, overlap=StaticVariable.overlap):
        """
        Generate tiles from the original image with specified overlap.
        Yields tuples of (tile, x0, y0, tile_id) where (x0, y0) is the top-left
        coordinate of the tile in the original image and tile_id is a unique identifier.
        """
        stride = int(tile_size * (1 - overlap))
        img_height, img_width, _ = original_image.shape
        for row_idx, y0 in enumerate(range(0, img_height, stride)):
            for col_idx, x0 in enumerate(range(0, img_width, stride)):
                tile_id = f"{row_idx}_{col_idx}"
                x1 = min(x0 + tile_size, img_width)
                y1 = min(y0 + tile_size, img_height)
                tile = original_image[y0:y1, x0:x1]
                yield tile, x0, y0, tile_id

    def polygon_to_bounding_box(json_data, file):
        bounding_box_list = []
        for item in json_data[file].values():
            if isinstance(item, dict):
                if item != {}:
                    length = len(item.values())
                    for i in range(length):
                        poly_x = item[str(i)]['shape_attributes']['all_points_x']
                        poly_y = item[str(i)]['shape_attributes']['all_points_y']
                        # Compute bounding box
                        x_min, x_max = min(poly_x), max(poly_x)
                        y_min, y_max = min(poly_y), max(poly_y)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        # Match format: [x, y, width, height]
                        bounding_box_list.append([int(x_min), int(y_min), int(bbox_width), int(bbox_height)])
        return bounding_box_list

    def get_coordinates_intersections(x_min, t_x_min, y_min, t_y_min, x_max, t_x_max, y_max, t_y_max):
        return (
            max(x_min, t_x_min),   # overlap left
            max(y_min, t_y_min),   # overlap top
            min(x_max, t_x_max),   # overlap right
            min(y_max, t_y_max)    # overlap bottom
        )

    def get_normalize_bounding_box(x_min, y_min, bbox_width, bbox_height, img_width, img_height):
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        w_norm = bbox_width / img_width
        h_norm = bbox_height / img_height
        return x_center, y_center, w_norm, h_norm

    def csv_data_to_annotations(csv_data, class_name):
        filtered = csv_data[csv_data["label_name"].isin(class_name)]

        labels = filtered["label_name"].tolist()

        bboxes = filtered[
            ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        ].values.tolist()

        return labels, bboxes
    def json_data_to_annotations(json_data, file):
        cluster_bboxes = Utils.polygon_to_bounding_box(json_data, file)
        cluster_labels = ["Cluster"] * len(cluster_bboxes)
        return cluster_labels, cluster_bboxes

    def adjust_bboxes_for_tile(annotations, x0, y0,
                               tile_size=StaticVariable.tile_size,
                               min_pixel_size=StaticVariable.min_pixel_size):
        has_annotation = False
        tile_bboxes = []
        tile_labels = []
        for annotation in annotations:
            label, bbox = annotation
            x_min, y_min, bbox_width, bbox_height = bbox
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            
            t_x_min, t_y_min = x0, y0
            t_x_max, t_y_max = x0 + tile_size, y0 + tile_size
            
            ix1, iy1, ix2, iy2 = Utils.get_coordinates_intersections(
                x_min, t_x_min, y_min, t_y_min, x_max, t_x_max, y_max, t_y_max
            )

            if ix1 < ix2 and iy1 < iy2:
                new_width = ix2 - ix1
                new_height = iy2 - iy1
                if new_width >= min_pixel_size and new_height >= min_pixel_size:
                    new_x = ix1 - x0
                    new_y = iy1 - y0
                    tile_bboxes.append((new_x, new_y, new_width, new_height))
                    tile_labels.append(label)
                    has_annotation = True
        return tile_labels, tile_bboxes, has_annotation

    # def get_bboxes_and_labels(image_path, file):
    #     csv_path, json_path  = Utils.replace(image_path)
    #     csv_data = Utils.get_csv_data(csv_path, file)
    #     json_data = Utils.get_json_data(json_path)
    #     thyrocyte_labels, thyrocyte_bboxes = Utils.csv_data_to_annotations(csv_data)
    #     cluster_labels, cluster_bboxes = Utils.json_data_to_annotations(json_data, file)
    #     return thyrocyte_labels, thyrocyte_bboxes, cluster_labels, cluster_bboxes

    def get_bboxes_and_labels_from_paths(
        initial_thyrocyte_path, 
        initial_cluster_path,
        updated_cluster_path,
        updated_thyrocyte_and_cluster_path,
        file
    ):
        thyrocyte_labels, thyrocyte_bboxes = [], []
        cluster_labels, cluster_bboxes = [], []
        # Prioritize updated annotations if available
        if isinstance(updated_thyrocyte_and_cluster_path, str) and updated_thyrocyte_and_cluster_path is not None:
            thyrocyte_and_cluster_data = Utils.get_csv_data(updated_thyrocyte_and_cluster_path, file)
            thyrocyte_labels, thyrocyte_bboxes = Utils.csv_data_to_annotations(thyrocyte_and_cluster_data, class_name=["Thyrocyte", "Thyrocytes", "Thycocyte"])
            cluster_labels, cluster_bboxes = Utils.csv_data_to_annotations(thyrocyte_and_cluster_data, class_name=["Cluster", "Clusters"])
        else:
            # Use initial thyrocyte annotations
            thyrocyte_data = Utils.get_csv_data(initial_thyrocyte_path, file)
            thyrocyte_labels, thyrocyte_bboxes = Utils.csv_data_to_annotations(thyrocyte_data, class_name=["Thyrocyte", "Thyrocytes", "Thycocyte"])
            # Use updated cluster annotations if available
            if isinstance(updated_cluster_path, str) and updated_cluster_path is not None:
                cluster_data = Utils.get_csv_data(updated_cluster_path, file)
                cluster_labels, cluster_bboxes = Utils.csv_data_to_annotations(cluster_data, class_name=["Cluster", "Clusters"])
            # Use initial cluster annotations otherwise
            elif isinstance(initial_cluster_path, str) and initial_cluster_path is not None:
                # Polygon to bounding box conversion from JSON
                cluster_data = Utils.get_json_data(initial_cluster_path)
                cluster_labels, cluster_bboxes = Utils.json_data_to_annotations(cluster_data, file)
                
        return thyrocyte_labels, thyrocyte_bboxes, cluster_labels, cluster_bboxes

        
    @staticmethod
    def handle_data_count_summary(invalid):
        for image_path, file in Utils.helper_os_walk():
            if image_path not in invalid:
                try:
                    csv_path, json_path  = Utils.replace(image_path)
                    csv_data = Utils.get_csv_data(csv_path, file)
                    # json_data = Utils.get_json_data(json_path)
                    thyrocytes = csv_data['label_name'].count()
                    # clusters = sum(len(item) for item in json_data[file].values() if isinstance(item, dict) and item)
                    # yield file, thyrocytes, clusters
                    # print(file.split('.')[0])
                    # print(Utils.column_look_up(file.split('.')[0]))
                    column = Utils.column_look_up(file.split('.')[0])
                    column = column.replace(" ", "_")  # Clean up column name
                    if column is None:
                        continue
                    yield file, thyrocytes, column
                except KeyError as e:
                    print(f"{e} in {file}")
            else:
                print(f"Skipping invalid file: {file}")
                    
    @staticmethod
    def column_look_up(target):
        df = StaticVariable.levels
        mask = df.eq(target)

        matched_cols = mask.any(axis=0)
        cols = matched_cols[matched_cols].index.tolist()

        # assert len(cols) <= 1, f"{target} appears in multiple columns!"

        return cols[0] if cols else None

    @staticmethod
    def data_split_csv(invalid):
        
        rows = [
            # {'File': file, 'Thyrocytes_Count': thyrocytes, 'Clusters_Count': clusters}
            {'File': file, 'Thyrocytes_Count': thyrocytes, 'Classification': classification}
            # for file, thyrocytes, clusters in Utils.handle_data_count_summary(invalid)
            for file, thyrocytes, classification in Utils.handle_data_count_summary(invalid)
        ]
        summary_df = pd.DataFrame(rows, columns=['File', 'Thyrocytes_Count', 'Classification'])
        summary_df.to_csv('results/dataset_summary.csv', index=False)
        
        # summary = pd.read_csv("/workspace/Special_Problem/dataset_summary.csv")
        # summary["Cluster_Group"] = summary["Clusters_Count"].apply(StaticVariable.cluster_group)
        
        # Stratified split (80% train, 10% val, 10% test)
        train_df, temp_df = train_test_split(
            # summary, test_size=0.2, stratify=summary["Cluster_Group"], random_state=42
            summary_df, test_size=0.2, stratify=summary_df["Classification"], random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["Classification"], random_state=42
        )
        
        train_df.to_csv('results/train_df_summary.csv', index=False)
        val_df.to_csv('results/val_df_summary.csv', index=False)
        test_df.to_csv('results/test_df_summary.csv', index=False)      
          
    def helper_os_walk(file_path=StaticVariable.data_path):
        for root, _, files in os.walk(file_path):
            for file in files:
                format = os.path.splitext(file)[1]
                if StaticVariable.is_supported(format):
                    image_path = os.path.join(root, file)
                    yield image_path, file

    def filter_less_than_eight_pixels(thyrocyte_bboxes, thyrocyte_labels):
        filtered = [
            (bbox, label)
            for bbox, label in zip(thyrocyte_bboxes, thyrocyte_labels)
            if bbox[2] > 8 and bbox[3] > 8
        ]
        
        if filtered:
            thyrocyte_bboxes, thyrocyte_labels = map(list, zip(*filtered))
        else:
            thyrocyte_bboxes, thyrocyte_labels = [], []
        return thyrocyte_bboxes, thyrocyte_labels

    def filter_and_split_mislabelled(thyrocyte_labels, thyrocyte_bboxes):
        thyrocyte_labels = np.array(thyrocyte_labels)
        thyrocyte_bboxes = np.array(thyrocyte_bboxes)
        # Filter masks
        cluster_mask = (thyrocyte_labels == "Cluster") | (thyrocyte_labels == "Clusters")
        thyrocyte_mask = ~cluster_mask
        # Separate bounding boxes and labels
        c_bboxes = thyrocyte_bboxes[cluster_mask].tolist()
        c_labels = thyrocyte_labels[cluster_mask].tolist()
        t_bboxes = thyrocyte_bboxes[thyrocyte_mask].tolist()
        t_labels = thyrocyte_labels[thyrocyte_mask].tolist()
        return t_bboxes, t_labels, c_bboxes, c_labels
    
    def get_specified_label_bboxes(thyrocyte_bboxes, thyrocyte_labels, cluster_bboxes, cluster_labels, specified_label):
        if specified_label == "Thyrocyte":
            return thyrocyte_bboxes, thyrocyte_labels
        elif specified_label == "Cluster":
            return cluster_bboxes, cluster_labels
        elif specified_label == "Both":
            bboxes = thyrocyte_bboxes + cluster_bboxes
            labels = thyrocyte_labels + cluster_labels
        else:
            raise ValueError("Invalid specified_label. Choose from 'Thyrocyte', 'Cluster', or 'Both'.")
        return bboxes, labels
    
    def getTransform(file_name):
        level = Utils.get_corresponding_level(file_name)
        if level == "LEVEL_I":
            transform = StaticVariable.transform_level_1
        elif level == "LEVEL_II":
            transform = StaticVariable.transform_level_2
        elif level ==  "LEVEL_III":
            transform = StaticVariable.transform_level_3_to_4
        elif level == "LEVEL_IV":
            transform = StaticVariable.transform_level_3_to_4
        elif level == "LEVEL_V":
            transform = StaticVariable.transform_level_5
        return transform
    
    @staticmethod
    def preprocess_original_image_annotations_generator(invalid, preprocess_callback=None, file_callback=None, label="Both"):
        df = StaticVariable.data_and_paths
        invalid_files = [
            file.split('/')[-1] for file in invalid
        ]
        for _, row in df.iterrows():
            file = row['File']
            if file not in invalid_files: # and file in StaticVariable.test_list:
                thyrocyte_path = row['Thyrocyte_Annotation_Path']
                cluster_path = row['Cluster_Annotation_Path']
                updated_cluster_path = row['Updated_Cluster_Annotation_Path']
                updated_thyrocyte_and_cluster_path = row['Updated_Thyrocyte_and_Cluster_Annotation_Path']
                image_path = row['Image_Path']
                thyrocyte_labels, thyrocyte_bboxes, cluster_labels, cluster_bboxes = Utils.get_bboxes_and_labels_from_paths(
                    thyrocyte_path, 
                    cluster_path,
                    updated_cluster_path,
                    updated_thyrocyte_and_cluster_path,
                    file
                    )
                thyrocyte_bboxes, thyrocyte_labels = Utils.filter_less_than_eight_pixels(thyrocyte_bboxes, thyrocyte_labels)
                
                if np.any((np.array(thyrocyte_labels) == "Cluster") | (np.array(thyrocyte_labels) == "Clusters")):
                    warnings.warn(f"[INFO] Mislabelled clusters found in thyrocyte annotations for file: {file}. Splitting them out.")
                    t_bboxes, t_labels, c_bboxes, c_labels = Utils.filter_and_split_mislabelled(thyrocyte_labels, thyrocyte_bboxes)
                    thyrocyte_bboxes, thyrocyte_labels = t_bboxes, t_labels
                    cluster_bboxes += c_bboxes
                    cluster_labels += c_labels
                    
                original_bboxes, original_labels = Utils.get_specified_label_bboxes(
                    thyrocyte_bboxes, thyrocyte_labels, cluster_bboxes, cluster_labels, specified_label=label
                )
                
                rgb_image = Utils.get_image_data(image_path)
                
                if file_callback is not None:
                    file_callback(file)
                try:
                    if preprocess_callback is not None and file in StaticVariable.train_list:
                        augmented_image, augmented_bboxes, augmented_labels = preprocess_callback(
                            rgb_image, original_bboxes, original_labels, Utils.getTransform(file)
                            )
                        yield "Augmented", (augmented_image, augmented_bboxes, augmented_labels)
                    yield "Original", (rgb_image, original_bboxes, original_labels)
                except Exception as e:
                    warnings.warn(f"Error processing file {file}: {e}")
                    ERROR.add(file)
            else:
                warnings.warn(f"Preprocessing skipped for invalid file: {file}")
    
    @staticmethod
    def preprocess_augmented_image_annotations_helper(rgb_image, original_bboxes, original_labels, transform):
        augmented = transform(image=rgb_image, bboxes=original_bboxes, labels=original_labels)
        return augmented['image'], augmented['bboxes'], augmented['labels']
 
    
    def get_corresponding_level(file):
        df = StaticVariable.summarized_levels
        row = df.loc[df["File"] == file, "Classification"]
        assert len(row) <= 1, "Duplicate File entries detected!"
        return row.iloc[0] if not row.empty else None
 
    def get_corresponding_actual_path(file):
        if file in StaticVariable.train_list:
            return StaticVariable.actual_train_image_path, StaticVariable.actual_train_label_path
        elif file in StaticVariable.val_list:
            return StaticVariable.actual_valid_image_path, StaticVariable.actual_valid_label_path
        elif file in StaticVariable.test_list:
            return StaticVariable.actual_test_image_path, StaticVariable.actual_test_label_path
        else:
            return None, None
    
    def get_corresponding_tiled_path(file):
        if file in StaticVariable.train_list:
            return StaticVariable.tile_train_image_path, StaticVariable.tile_train_label_path
        elif file in StaticVariable.val_list:
            return StaticVariable.tile_valid_image_path, StaticVariable.tile_valid_label_path
        elif file in StaticVariable.test_list:
            return StaticVariable.tile_test_image_path, StaticVariable.tile_test_label_path
        else:
            return None, None
            
    def write_annotations(image, bboxes, labels, output_path):
        img_height, img_width, _ = image.shape
        
        # Clear file before writing (avoid leftover content)
        open(output_path, 'w').close()
        
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, bbox_width, bbox_height = bbox
            class_id = StaticVariable.label_map[label]
            Utils.normalize_bounding_box(
                x_min, y_min, bbox_width, bbox_height, img_height, img_width,
                output_path, class_id
            )
            
    def normalize_bounding_box(
        x_min, y_min, bbox_width, bbox_height, 
        img_height, img_width, output_path, class_id
    ):
        x_center, y_center, w_norm, h_norm = Utils.get_normalize_bounding_box(
            x_min, y_min, bbox_width, bbox_height, img_width, img_height
        )
        with open(output_path, 'a') as f:
            f.write(f"{class_id} {x_center:.8f} {y_center:.8f} {w_norm:.8f} {h_norm:.8f}\n")
   
    @staticmethod
    def process_tile_generator(data, prefix):
        image, bboxes, labels = data
        annotations = list(zip(labels, bboxes)) # stores the entire list in memory, reusable
        """
        label, bbox = annotation
        """
        for tile, x0, y0, tile_id in Utils.image_tiling(image):
            tile_labels, tile_bboxes, has_annotation = Utils.adjust_bboxes_for_tile(annotations, x0, y0)
            if has_annotation:
                # if (prefix == "augmented" and ("Cluster" in tile_labels or "Clusters" in tile_labels)) or prefix == "original":
                yield (tile, tile_bboxes, tile_labels), tile_id
    
    def image_save_kwargs(format):
        save_kwargs = {}
        if format in [".jpg", ".jpeg"]:
            """ subsampling=0 ensures no chroma subsampling for better quality. 
            Prevents chroma subsampling. Preserves fine color detail (important in cytology)"""
            save_kwargs = dict(quality=95, subsampling=0)
        elif format == ".png":
            save_kwargs = dict(optimize=True)
        return save_kwargs
    
    def save_data(data, image_path, label_path, prefix, file, level):
        
        assert level is not None, "Level is None — invalid label"
        assert isinstance(level, str), f"Invalid level type: {type(level)}"
        
        image, bboxes, labels = data
        
        image = Utils.pad_image(image)
        pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
        
        
        
        # Image directory
        img_dir = os.path.join(image_path, level)
        os.makedirs(img_dir, exist_ok=True)
        
        # Label directory
        lbl_dir = os.path.join(label_path, level)
        os.makedirs(lbl_dir, exist_ok=True)
        
        # Save image
        format = os.path.splitext(file)[1]
        save_kwargs = Utils.image_save_kwargs(format)
        
        pil_img.save(
            os.path.join(img_dir, f"{prefix}_{file}"),
            **save_kwargs
        )
        
        # Save labels
        label_file = os.path.join(
            lbl_dir,
            f"{prefix}_{os.path.splitext(file)[0]}.txt"
        )
        
        Utils.write_annotations(image, bboxes, labels, label_file)
        
    def pad_image(image):
        img_height, img_width, _ = image.shape
        desired = StaticVariable.tile_size
        
        pad_bottom = max(desired - img_height, 0)
        pad_right = max(desired - img_width, 0)

        if pad_bottom > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(
                image,
                0, pad_bottom, 0, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=StaticVariable.value
            )
        return image
    
    def cluster_exist(content):
        labels = list(content.split('\n'))
        for label in labels:
            if label.split(' ')[0] == '0':
                return True
        return False
    
    def get_datatype(file):
        if file in StaticVariable.train_list:
            return 'train' 
        if file in StaticVariable.val_list:
            return 'val'
        if file in StaticVariable.test_list:
            return 'test'
    
    def get_file(file):
        file = file.split('/')[-1]
        file_name = re.search(r'LS-(\d+)', file)
        return file_name.group(0) + "." + file.split('.')[-1]
    
    def copy_tiles_and_labels(df, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(dest_dir.replace("images", "labels"), exist_ok=True)

        for _, row in df.iterrows():
            tile_path = row['tile_path']
            label_path = row['label_path']
            
            # Copy image
            if os.path.exists(tile_path):
                shutil.copy(tile_path, os.path.join(dest_dir, os.path.basename(tile_path)))
            
            # Copy label
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_dir.replace("images", "labels"), os.path.basename(label_path)))

    def saved_original_images_for_visualization(data, name, file):
        image, bboxes, labels = data

        # Convert numpy (H,W,C) to a PIL RGB image
        pil_img = Image.fromarray(image.astype('uint8')).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        for bbox, label in zip(bboxes, labels):
            x_min, y_min, box_width, box_height = bbox

            # Compute x_max, y_max for drawing
            x_max = x_min + box_width
            y_max = y_min + box_height

            color = "red" if label == 'Cluster' or label == 'Clusters' else "black"
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            # Draw text label
            draw.text((x_min, y_min), str(label), fill=color)

        
        format = os.path.splitext(file)[1]
        folder = os.path.splitext(file)[0]
        save_kwargs = Utils.image_save_kwargs(format)
        # conf_values = ['confidence_value 0.5', 'confidence_value 0.35', 'confidence_value 0.65']
        # for conf_value in conf_values:
        dir = f'results/test_data_for_validation/'
        # print(folder)
        os.makedirs(dir, exist_ok=True)
        pil_img.save(
            f"{dir}/{name}",
            **save_kwargs
        )
        
    def iter_annotation_paths():
        no_cluster_files = StaticVariable.no_cluster_files
        no_cluster_files = (
            no_cluster_files[no_cluster_files["remarks"] == "no cluster found"]["File"]
            .astype(str)
            .tolist()
        )
        for image_path, file in Utils.helper_os_walk():
            file_name = os.path.splitext(file)[0]
            
            thyrocyte_path, cluster_path = Utils.replace(image_path)
            
            # Force cluster to None if explicitly marked
            if file_name in no_cluster_files:
                cluster_path = None
                
            # Updated annotations
            updated_cluster_annotation_path, _ = Utils.replace(
                image_path,
                affixes=[" - UPDATED ANNOTATIONS", "-cluster.csv"],
            )

            updated_thyrocyte_and_cluster_annotation_path = Utils.updated_replace(
                image_path, file_name
            )
            
            # Validate paths
            if not thyrocyte_path or not os.path.exists(thyrocyte_path):
                thyrocyte_path = None

            if not cluster_path or not os.path.exists(cluster_path):
                cluster_path = None

            if (
                not updated_cluster_annotation_path
                or not os.path.exists(updated_cluster_annotation_path)
            ):
                updated_cluster_annotation_path = None

            if (
                not updated_thyrocyte_and_cluster_annotation_path
                or not os.path.exists(updated_thyrocyte_and_cluster_annotation_path)
            ):
                updated_thyrocyte_and_cluster_annotation_path = None

            yield (
                file,
                thyrocyte_path,
                cluster_path,
                updated_cluster_annotation_path,
                updated_thyrocyte_and_cluster_annotation_path,
                image_path,
            )
            
    def save_data_path_to_csv(
        output_csv="results/explore_data_annotation_paths.csv",
    ):
        rows = [
                {
                    "File": file,
                    "Thyrocyte_Annotation_Path": thyrocyte_path,
                    "Cluster_Annotation_Path": cluster_path,
                    "Updated_Cluster_Annotation_Path": updated_cluster_annotation_path,
                    "Updated_Thyrocyte_and_Cluster_Annotation_Path": updated_thyrocyte_and_cluster_annotation_path,
                    "Image_Path": image_path,
                }
                for (
                    file,
                    thyrocyte_path,
                    cluster_path,
                    updated_cluster_annotation_path,
                    updated_thyrocyte_and_cluster_annotation_path,
                    image_path,
                )
                in Utils.iter_annotation_paths()
            ]
        df = pd.DataFrame(
            rows,
            columns=[
                "File",
                "Thyrocyte_Annotation_Path",
                "Cluster_Annotation_Path",
                "Updated_Cluster_Annotation_Path",
                "Updated_Thyrocyte_and_Cluster_Annotation_Path",
                "Image_Path",
            ],
        )
        df.to_csv(output_csv, index=False)
    
class CallbackUtil:
    def __init__(self):
        self.file = None
    
    def set_file(self, file):
        self.file = file
    
    def get_file(self):
        return self.file

if __name__ == '__main__':
    
    for image_path, file in Utils.helper_os_walk():
        print(image_path)
    
    # Utils.save_data_path_to_csv()
    # not_classified = []
    
    # for dir in StaticVariable.DIR_PATH:
    #     os.makedirs(dir, exist_ok=True)
        
    # callback = CallbackUtil()
    # invalid = Utils.check_dataset()
    
    # for i in invalid:
    #     print(f"Invalid dataset found: {i}")
        
    # Utils.data_split_csv(invalid)
        
    # for data_type, data in Utils.preprocess_original_image_annotations_generator(
    #     invalid, 
    #     Utils.preprocess_augmented_image_annotations_helper,
    #     callback.set_file,
    #     label="Thyrocyte"
    # ):
    #     file = callback.get_file()
    #     level = Utils.get_corresponding_level(file)
    #     prefix = "augmented" if data_type == "Augmented" else "original"
        
    #     # """Save Original Images or Augmented Images for Visualization"""
    #     # if prefix == 'original': # and level != 'LEVEL_IV' and level != 'LEVEL_V':
    #     #     # print(f"Saving visualization for {file}...")
    #     #     Utils.saved_original_images_for_visualization(data, f"Ground_Truth_{file}", f'original_{file}')
        
    #     """ For Original Image | Untiled Image """
    #     image_path, label_path = Utils.get_corresponding_actual_path(file)
        
    #     """ Handle files not classified into any dataset split """
    #     if image_path is None or label_path is None:
    #         print(f"Skipping file {file} as it does not belong to any dataset split.")
    #         not_classified.append(file)
    #         continue
        
    #     # Utils.save_data(data, image_path, label_path, prefix, file, level)
        
    #     # if level in ['LEVEL_I', 'LEVEL_II','LEVEL_III']:
    #     """ Tiling """
    #     # image_path, label_path = Utils.get_corresponding_tiled_path(file)
    #     for tile_data, tile_id in Utils.process_tile_generator(data, prefix):
    #         file_tile = file.replace(".", f"_{tile_id}.") 
    #         Utils.save_data(tile_data, image_path, label_path, prefix, file_tile, level)
            
    #         # Utils.saved_original_images_for_visualization(tile_data, f"{level}_{prefix}_{file_tile}")
    #         # Utils.saved_original_images_for_visualization(tile_data, level , f"{prefix}_{file_tile}")
    #         break
        
    # print("Errors found in files: ")
    # sorted_errors = sorted(ERROR)
    # print(sorted_errors)
    
    # print("Files not classified into any dataset split:")
    # not_classified_df = pd.DataFrame(not_classified, columns=['Not Classified'])
    # not_classified_df.to_csv('/workspace/Special_Problem/not_classified_df.csv', index=False)