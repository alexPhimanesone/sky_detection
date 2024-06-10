git clone

python -3.9 -m venv sky_detection_env

pip install numpy
pip install pyyaml

pip3 install pytest-runner --upgrade
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U openmim
mim install mmengine==0.10.3
mim install mmcv==2.1.0
pip install mmsegmentation==1.2.2

pip install -r <path_to_fisheye_to_equirectangular_requirements>

créer un dossier data dans le répertoire racine
y ajouter model.pth
