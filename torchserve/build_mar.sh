tar -cvzf files.tar.gz model/ models/ utils/ base/ logger/ parse_config.py
torch-model-archiver --model-name age-gender --version 1.0 --handler handler.py \
                     --serialized-file blank.pth -r requirements.txt --extra-files files.tar.gz -f
sudo cp age-gender.mar /mnt/docker/volumes/model-servers/_data/pytorch-models/