from PIL import Image
import argparse
import os
import hashlib
import re
import shutil
import glob
import numpy as np
import csv

def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = '07d'):

    if not os.path.isdir(input_dir):
        raise ValueError(f'{input_dir} is not an existing directory')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #clear log_file
    with open(log_file, 'w') as log:
        log.write('') 
    
    input_dir = os.path.abspath(input_dir)
    all_files = []
    all_paths = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)

    for path in all_paths:
        if os.path.isfile(path):
            all_files.append(path)

    all_files.sort()
    all_copied_files = []
    all_copied_labels = []
    counter = 0

    for valid_file in all_files:

        try:
            valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG']
            extension = os.path.splitext(valid_file)[1]
            #print(extension)

            if extension not in valid_extensions:
                raise ValueError(1)
            
            if os.path.getsize(valid_file) > 250000:
                raise ValueError(2)

            try:
                img = Image.open(valid_file)
                img.verify()
                img = Image.open(valid_file)

            except:
                raise ValueError(3)

            if img.mode != 'RGB' and img.mode != 'L':
                raise ValueError(4)

            if len(img.size) < 2 or img.size[0] < 100 or img.size[1] < 100:
                raise ValueError(4)

            if np.array(img).var() <= 0:
                raise ValueError(5)

            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in all_copied_files:
                raise ValueError(6)

            #if pass, all rules correct, copy valid_files
            #obtain label from file.name
            label = os.path.splitext(os.path.basename(valid_file))[0]
            #remove the digits from the label
            label = re.sub(r'\d', '', label)

            all_copied_files.append(img_hash)
            all_copied_labels.append(label)

            shutil.copyfile(valid_file, os.path.join(output_dir, f'{counter:{formatter}}{extension}'))

            counter += 1

        except ValueError as e:
            with open(log_file, 'a') as log:
                log.write(f'{os.path.relpath(valid_file, input_dir)},{e.args[0]}\n')

    #create labels.csv
    with open(os.path.join(output_dir, 'labels.csv'), 'w', newline='') as csv_file:
        headers = ['name', 'label']
        write_csv = csv.DictWriter(csv_file, headers, delimiter=';')
        write_csv.writeheader()
        for i, label in enumerate(all_copied_labels):
            write_csv.writerow({'name': f'{i:{formatter}}{extension}', 'label': label})

    return counter

if __name__ == '__main__':
    rules = """Rules:

1. The file name ends with .jpg, .JPG, .jpeg or .JPEG.
2. The file size does not exceed 250kB (=250 000 Bytes).
3. The file can be read as image (i.e., the PIL/pillow module does not raise an exception
   when reading the file).
4. The image data has a shape of (H, W, 3) with H (height) and W (width) larger than or
   equal to 100 pixels, and the three channels must be in the order RGB (red, green, blue).
   Alternatively, the image can also be grayscale and have a shape of only (H, W) with the
   same width and height restrictions.
5. The image data has a variance larger than 0, i.e., there is not just one common pixel in
   the image data.
6. The same image has not been copied already."""
    
    parser = argparse.ArgumentParser(description=rules, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_dir', type=str, help='input_dir')
    parser.add_argument('output_dir', type=str, help='output_dir')
    parser.add_argument('log_file', type=str, help='log_file')
    parser.add_argument('--formatter', type=str, default='07d', help='formatter')
    args = parser.parse_args()

    validate_images(args.input_dir, args.output_dir, args.log_file, args.formatter)
    num_files_copied  = validate_images(args.input_dir, args.output_dir, args.log_file, args.formatter)
    print(f'\nFiles copied: {num_files_copied}')

### python image_validator.py /home/ari/second_semester/python_2/As_1/handy_images_resized /home/ari/second_semester/python_2/As_1/test log_file_test
    
