import os
import zipfile
import subprocess

def download_datasets(destination_dir):
    # Specify the competition name
    competition_name = "histopathologic-cancer-detection"

    # Construct the command
    command = f"kaggle competitions download -c {competition_name} -p {destination_dir}"

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
        print("Dataset has been downloaded to:", destination_dir)
        # Unzip the downloaded file
        zip_file_path = os.path.join(destination_dir, f"{competition_name}.zip")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        print("Dataset has been extracted.")
        
        # Remove the zip file
        os.remove(zip_file_path)
        print("Zip file has been deleted.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

#if __name__ == "__main__":
#    data_dir = os.path.join(os.getcwd(), 'dataset')
#    print(f'data_dir: {data_dir}')
    # Make sure that the destination_dir is available
#    if not os.path.exists(data_dir):
#        os.mkdir(data_dir)
    # Download the dataset
#    download_datasets(data_dir)
