# SOVA ASR

SOVA ASR is a fast speech recognition solution based on [Wav2Letter](https://arxiv.org/abs/1609.03193) architecture. It is designed as a REST API service and it can be customized (both code and models) for your needs.

## Installation

The easiest way to deploy the service is via docker-compose, so you have to install Docker and docker-compose first. Here's a brief instruction for Ubuntu:

#### Docker installation

*	Install Docker:
```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
$ sudo usermod -aG docker $(whoami)
```
In order to run docker commands without sudo you might need to relogin.
*   Install docker-compose:
```
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```

*   (Optional) If you're planning on using CUDA run these commands:
```
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
$ sudo apt-get update
$ sudo apt-get install nvidia-container-runtime
```
Add the following content to the file **/etc/docker/daemon.json**:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
Restart the service:
```bash
$ sudo systemctl restart docker.service
``` 

#### Build and deploy

**In order to run service with pretrained models you will have to download http://dataset.sova.ai/SOVA-ASR/Data.tar.**

*   Clone the repository, download the pretrained models archive and extract the contents into the project folder:
```bash
$ git clone --recursive https://github.com/sovaai/sova-asr.git
$ cd sova-asr/
$ wget http://dataset.sova.ai/SOVA-ASR/Data.tar
$ tar -xvf Data.tar && rm Data.tar
```

*   Build docker image
     *   If you're planning on using GPU (it is required for training and can be used for inference): build *sova-asr* image using the following command:
     ```bash
     $ sudo docker-compose build sova-asr
     ```
     *   If you're planning on using CPU only: modify `Dockerfile`, `docker-compose.yml` (remove the runtime and environment sections) and `config.ini` (*cpu* should be set to 0) and build *sova-asr* image:
     ```bash
     $ sudo docker-compose build sova-asr
     ```

*	Run web service in a docker container
     ```bash
     $ sudo docker-compose up -d sova-asr
     ```

## Testing

To test the service you can send a POST request:
```bash
$ curl --request POST 'http://localhost:8888/asr' --form 'audio_blob=@"Data/test.wav"'
```

## Finetuning acoustic model

If you want to finetune the acoustic model you can set hyperparameters and paths to your own train and validation manifest files and run the training service.

*	Set training options in *Train* section of **config.ini**. Train and validation csv manifest files should contain comma-separated audio file paths and reference texts in each line. For instance:
     ```bash
     Data/Audio/000000.wav,добрый день
     Data/Audio/000001.wav,как ваши дела
     ...
     ```
*	Run training in docker container:
     ```bash
     $ sudo docker-compose up -d sova-asr-train
     ```

## Customizations

If you want to train your own acoustic model refer to [PuzzleLib tutorials](https://puzzlelib.org/tutorials/Wav2Letter/). Check [KenLM documentation](https://kheafield.com/code/kenlm/) for building your own language model. This repository was tested on Ubuntu 18.04 and has pre-built .so Trie decoder files for Python 3.6 running inside the Docker container, for modifications you can get your own .so files using [Wav2Letter++](https://github.com/facebookresearch/wav2letter) code for building Python bindings. Otherwise you can use a standard Greedy decoder (set in config.ini).
