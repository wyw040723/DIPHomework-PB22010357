import os
import urllib.request
import zipfile

url = 'https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-nocuda.zip'
root = os.path.abspath(os.path.dirname(__file__))
zip_path = os.path.join(root, 'colmap-x64-windows-nocuda.zip')
install_dir = os.path.join(root, 'colmap_nocuda')

print('Downloading COLMAP...')
urllib.request.urlretrieve(url, zip_path)
print('Downloaded to', zip_path)

os.makedirs(install_dir, exist_ok=True)
print('Extracting...')
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(install_dir)
print('Extracted to', install_dir)
print('Contents:', os.listdir(install_dir)[:20])
