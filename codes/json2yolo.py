import json
import os

def convert_to_yolo(json_file, output_directory):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []
    dict_img_id = {}

    for img in data['images']:
        file_name = img['file_name']
        #if file_name[5:7] in ['10','11','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']:
            #continue
        img_id = img['id']
        img_width = img['width']
        img_height = img['height']
        dict_img_id[img_id] = [file_name[:-4],img_width,img_height]

    for obj in data['annotations']:
        class_id = obj['category_id']
        x_min = obj['bbox'][0]
        y_min = obj['bbox'][1]
        box_width = obj['bbox'][2]
        box_height = obj['bbox'][3]
        img_id = obj['image_id']
        
        if img_id not in dict_img_id:
            continue
        # Convert to YOLO format
        image_width = dict_img_id[img_id][1]
        image_height = dict_img_id[img_id][2]
        center_x = (x_min + box_width / 2) / image_width
        center_y = (y_min + box_height / 2) / image_height
        norm_width = box_width / image_width
        norm_height = box_height / image_height
        
        yolo_annotations.append([f"{class_id} {center_x} {center_y} {norm_width} {norm_height}",img_id])

    for annotation in yolo_annotations:
        newpath = output_directory + dict_img_id[annotation[1]][0][:-10]
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        file_path = output_directory + dict_img_id[annotation[1]][0] + ".txt"
        try:
            with open(file_path, 'a') as f:
                f.write(annotation[0] + '\n')
        except:
            with open(file_path, 'w') as f:
                f.write(annotation[0] + '\n')       

if __name__ == '__main__':
    # Example usage
    convert_to_yolo('/home/umberto.pereira/Mosquitoes/dataset/v2/coco_json_folds/5folds/40m/coco_format_train3_tire.json', '/home/umberto.pereira/Mosquitoes/datasets/v2/labels/')
