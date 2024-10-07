import os
import argparse
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

# Define arrays for license plate character decoding
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# Function to parse the filename and extract annotations
def parse_ccpd_filename(filename):
    # Remove the .jpg extension
    filename = filename.replace('.jpg', '')
    
    # Split the filename into parts
    parts = filename.split('-')
    
    # Area and tilt degrees
    area = int(parts[0])  # Area ratio (not used in annotation)
    tilt_degrees = parts[1].split('_')
    horizontal_tilt = int(tilt_degrees[0])
    vertical_tilt = int(tilt_degrees[1])
    
    # Bounding box coordinates
    bbox_coords = parts[2].split('_')
    min_val = bbox_coords[0].split('&')
    max_val = bbox_coords[1].split('&')
    xmin, ymin = int(min_val[0]), int(min_val[1])
    xmax, ymax = int(max_val[0]), int(max_val[1])
    # print(xmin, ymin, xmax, ymax)
    
    # Four vertices coordinates (in the order: right-bottom, right-top, left-top, left-bottom)
    # vertices_coords = parts[3].split('_')
    # vertices = [i.split('&') for i in vertices_coords]
    
    # # License plate number
    # lp_indices = list(map(int, parts[4].split('_')))
    # license_plate = provinces[lp_indices[0]] + alphabets[lp_indices[1]] + ''.join([ads[i] for i in lp_indices[2:]])
    
    # # Brightness and blurriness (optional, can be used if needed)
    # brightness = int(parts[5])
    # blurriness = int(parts[6])
    
    # Construct the annotation dictionary
    annotation = {
        # 'area': area,
        # 'tilt': {
        #     'horizontal': horizontal_tilt,
        #     'vertical': vertical_tilt
        # },
        'bounding_box': {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        } #,
        # 'vertices': vertices,
        # 'license_plate': license_plate,
        # 'brightness': brightness,
        # 'blurriness': blurriness
    }
    
    return annotation

# Function to create Pascal VOC XML from the annotation
def create_voc_xml(annotation, filename, folder, image_size):
    # Create XML structure
    root = ET.Element('annotation')
    
    # Filename and folder
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = filename
    
    # Image size
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(image_size[0])
    ET.SubElement(size, 'height').text = str(image_size[1])
    ET.SubElement(size, 'depth').text = str(image_size[2])  # Assuming RGB (depth=3)
    
    # Object (license plate)
    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = 'license_plate'
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'
    
    # Bounding box
    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(annotation['bounding_box']['xmin'])
    ET.SubElement(bndbox, 'ymin').text = str(annotation['bounding_box']['ymin'])
    ET.SubElement(bndbox, 'xmax').text = str(annotation['bounding_box']['xmax'])
    ET.SubElement(bndbox, 'ymax').text = str(annotation['bounding_box']['ymax'])
    
    # Vertices
    # vertices = ET.SubElement(obj, 'vertices')
    # for i, (x, y) in enumerate(annotation['vertices']):
    #     ET.SubElement(vertices, f'vertex_{i}_x').text = str(x)
    #     ET.SubElement(vertices, f'vertex_{i}_y').text = str(y)
    
    # Beautify the XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    return xml_str

# Function to save the XML to file
def save_voc_xml(xml_str, output_path):
    with open(output_path, 'w') as f:
        f.write(xml_str)

# Function to get image size using PIL
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size[0], img.size[1], len(img.getbands())  # Width, height, and depth

# Function to process all files in the dataset and generate XML annotations
def process_dataset(ccpd_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through files in the CCPD folder
    for file in tqdm(os.listdir(ccpd_folder)):
        if file.endswith('.jpg'):
            # Parse the filename to get the annotation
            annotation = parse_ccpd_filename(file)
            
            # Get image size dynamically
            image_path = os.path.join(ccpd_folder, file)
            image_size = get_image_size(image_path)
            
            # Create VOC XML content
            xml_content = create_voc_xml(annotation, file, folder='JPEGImages', image_size=image_size)
            
            # Save the XML file
            xml_filename = file.replace('.jpg', '.xml')
            xml_output_path = os.path.join(output_folder, xml_filename)
            save_voc_xml(xml_content, xml_output_path)
            # print(f"Saved: {xml_output_path}")

# Main function to run the script with argparse
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CCPD dataset annotation to VOC XML format converter")
    parser.add_argument('--ccpd_folder', type=str, required=True, help="Path to the CCPD dataset folder containing images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder where XML files will be saved")

    # Parse the arguments
    args = parser.parse_args()

    # Process the dataset and generate VOC XML annotations
    process_dataset(args.ccpd_folder, args.output_folder)
