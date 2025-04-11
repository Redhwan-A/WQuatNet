# import gdown
#
# # url = "https://drive.google.com/file/d/1agFhy7QiMZWaB4q13_8W9jHeTy2-DyRx/view?usp=sharing"
# url = 'https://drive.google.com/file/d/1agFhy7QiMZWaB4q13_8W9jHeTy2-DyRx/view?usp=sharing'
# output = "RepVGG-D2se-200epochs-train.pth"
#
# gdown.download(url, output, use_cookies=False)
#
#


import gdown

# Correct URL after changing sharing permissions
url = 'https://drive.google.com/uc?export=download&id=1agFhy7QiMZWaB4q13_8W9jHeTy2-DyRx'
output = "RepVGG-D2se-200epochs-train.pth"

# Download the file
gdown.download(url, remaining_ok=True)


# import os
# import subprocess
#
# def download_file_from_google_drive(file_id, destination):
#     # URL for Google Drive download
#     base_url = "https://docs.google.com/uc?export=download"
#     download_url = f"{base_url}&id={file_id}"
#
#     # Using wget to download the file
#     # '-O' flag specifies the output file
#     subprocess.run(["wget", "--no-check-certificate", download_url, "-O", destination], check=True)
#     print(f"Download complete: {destination}")
#
# # Example usage
# file_id = "1agFhy7QiMZWaB4q13_8W9jHeTy2-DyRx"  # Replace with the actual file ID
# destination = "RepVGG-D2se-200epochs-train.pth"
#
# # Ensure the destination path is valid
# if not os.path.exists(destination):
#     print(f"Downloading {destination} from Google Drive...")
#     download_file_from_google_drive(file_id, destination)
# else:
#     print(f"File {destination} already exists.")

