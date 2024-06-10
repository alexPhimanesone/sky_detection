# Sky Detection Project - Setup Instructions

Follow these steps to set up the environment and install the necessary dependencies.

### Step 1: Clone the Repository

```sh
git clone https://github.com/alexPhimanesone/sky_detection.git
```

### Step 2: Create a virtual environment

```sh
python -3.9 -m venv sky_detection_env
```

### Step 3: Install dependencies
```sh
pip install numpy
pip install pyyaml
pip3 install pytest-runner --upgrade
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U openmim
mim install mmengine==0.10.3
mim install mmcv==2.1.0
pip install mmsegmentation==1.2.2
pip install -r <path_to_fisheye_to_equirectangular_requirements>
```

### Step 4: Create the data folder and add the model
Créer un dossier data dans le répertoire racine
Y ajouter model.pth