pip install --upgrade pip
pip install torch torchvision torchaudio torchao torchtune transformers datasets wandb tiktoken lm_eval boto3 -U
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja -U

mkdir temp
cd temp
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install .
pip install lm_eval["ruler"]
cd ..

conda install gxx_linux-64 -y
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install .
cd ../..
rm -rf ./temp
