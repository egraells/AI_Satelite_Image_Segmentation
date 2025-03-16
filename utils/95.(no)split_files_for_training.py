import os
import shutil
import random

def split_dataset(geotiff_folder, png_folder, train_ratio=0.85):
    """
    Splits correlated GeoTIFF and PNG files into training and testing sets.
    
    :param geotiff_folder: Folder containing GeoTIFF files named tile_X_Y.tif.
    :param png_folder: Folder containing PNG files named ttile_nonorm_X_Y.png.
    :param output_folder: Base output directory where 'training' and 'testing' folders will be created.
    :param train_ratio: Percentage of files allocated to training (default: 85%).
    """
    geotiff_training_folder = os.path.join(geotiff_folder, "training")
    geotiff_testing_folder = os.path.join(geotiff_folder, "testing")
    png_training_folder = os.path.join(png_folder, "training")
    png_testing_folder = os.path.join(png_folder, "testing")
    
    os.makedirs(geotiff_training_folder, exist_ok=True)
    os.makedirs(geotiff_testing_folder, exist_ok=True)
    os.makedirs(png_training_folder, exist_ok=True)
    os.makedirs(png_testing_folder, exist_ok=True)
    
    geotiff_files = [f for f in os.listdir(geotiff_folder) if f.endswith(".tif") and "novalues" not in f]
    random.shuffle(geotiff_files)
    split_index = int(len(geotiff_files) * train_ratio)
    geotiff_training_set = geotiff_files[:split_index]
    geotiff_testing_set = geotiff_files[split_index:]
    
    png_files = set(os.listdir(png_folder))
    
    log_file = os.path.join(".", "missing_files.log")

    with open(log_file, "w") as log:
        for dataset, dataset_folder in [(geotiff_training_set, geotiff_training_folder), (geotiff_testing_set, geotiff_testing_folder)]:
            for geotiff in dataset:
                x, y = geotiff.split("_")[1], geotiff.split("_")[2].split(".")[0]
                png_filename = f"tile_nonorm_{x}_{y}.png"
                
                geotiff_src = os.path.join(geotiff_folder, geotiff)
                geotiff_dest = os.path.join(dataset_folder, geotiff)
                
                png_src = os.path.join(png_folder, png_filename)

                folder_type = dataset_folder.split(os.sep)[-1] # training or testing
                png_dest = os.path.join(png_folder, folder_type)
                png_dest = os.path.join(png_dest, png_filename)
                
                if png_filename in png_files:
                    shutil.copyfile(geotiff_src, geotiff_dest)
                    shutil.copyfile(png_src, png_dest)
                else:
                    log.write(f"Missing PNG for {geotiff}: {png_filename}\n")
                    print(f"Warning: Missing PNG for {geotiff}")
    
    print("Dataset split completed.")

if __name__ == "__main__":
    geotiff_folder = "D:\\tiles_training\\tiles_masks\\tiffs"  
    png_folder =     "D:\\tiles_training\\sentinel2april\\tile_no_norm"  # cal modificar per cada execució
    
    split_dataset(geotiff_folder, png_folder, 0.85)
