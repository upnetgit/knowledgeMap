source ./myenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_all.txt
pip install faster-whisper
pip install -U fvcore iopath omegaconf pyyaml
pip install ctranslate2
pip install tesseract
sudo apt update
sudo apt install -y ffmpeg
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim ffmpeg
tesseract --list-langs | grep chi_sim

export NEO4J_URI=''
export NEO4J_USERNAME=''
export NEO4J_PASSWORD=''
export NEO4J_DATABASE=''
export AURA_INSTANCEID=''
export AURA_INSTANCENAME='Free instance'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python build_kg.py \
  --data-dir ./data \
  --output-dir ./kg_output \
  --use-xmodaler-video \
  --xmodaler-model-type tdconved \
  --video-preprocess \
  --video-preprocess-fix \
  --video-fix-mode inplace \
  --language zh \
  --device-mode cuda

export ENABLE_MANUAL_ANNOTATION=true
python app.py
python tools/import_manual_annotations.py \
  --annotation-file kg_output/manual_video_annotations.jsonl
