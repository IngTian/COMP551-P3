import shutil
import os
import subprocess
import json
from pathlib import Path

# Install all python dependencies.
print("INSTALLING DEPENDENCIES...")
subprocess.call(['pip', 'install', '-r', 'requirements.txt'], stdout=subprocess.DEVNULL)

if shutil.which("kaggle") is not None:
    subprocess.call(['pip', 'install', '--user', 'kaggle'], stdout=subprocess.DEVNULL)

# Create necessary folders.
print("CREATING NECESSARY FOLDERS...")
if not os.path.exists("dataset"):
    os.mkdir("dataset")
if not os.path.exists("out"):
    os.mkdir("out")

# Downloading datasets.

# Install kaggle configurations.
credential = {"username": "ingtian", "key": "27f85d4210d3ec4258ff953b1966e613"}
if not os.path.exists(Path.home() / ".kaggle"):
    os.mkdir(Path.home() / ".kaggle")
with open(Path.home() / ".kaggle" / "kaggle.json", 'w') as kaggle_config:
    json.dump(credential, kaggle_config)

print("DOWNLOADING DATASETS...")
subprocess.call(
    ["kaggle", "competitions", "download", "-p", "dataset", "-c", "comp-551-fall-2021"],
    stdout=subprocess.DEVNULL
)

# Unzip and prepare files.
print("PREPARE DATASETS...")

# Linux.
if os.path.exists("./dataset/images_l.pkl.zip"):
    shutil.unpack_archive("./dataset/images_l.pkl.zip", "./dataset")
    shutil.unpack_archive("./dataset/images_ul.pkl.zip", "./dataset")
    shutil.unpack_archive("./dataset/images_test.pkl.zip", "./dataset")
    shutil.unpack_archive("./dataset/labels_l.pkl.zip", "./dataset")
    os.remove("./dataset/images_l.pkl.zip")
    os.remove("./dataset/images_ul.pkl.zip")
    os.remove("./dataset/images_test.pkl.zip")
    os.remove("./dataset/labels_l.pkl.zip")
# MacOS
else:
    shutil.unpack_archive("./dataset/comp-551-fall-2021.zip", "./dataset")
    os.remove("./dataset/comp-551-fall-2021.zip")

print("DONE(*¯︶¯*)")
