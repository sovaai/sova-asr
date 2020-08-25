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

*   Clone the repository, download the pretrained models archive and extract the contents into the project folder:
```bash
$ git clone --recursive https://github.com/sovaai/sova-asr.git
$ cd sova-asr/
$ wget http://dataset.sova.ai/SOVA-ASR/Data.tar
$ tar -xvf Data.tar && rm Data.tar
```

*   Build docker image
     *   Build *sova-asr-gpu* image if you're planning on using GPU:
     ```bash
     $ sudo docker-compose build sova-asr-gpu
     ```
     *   Build *sova-asr* image if you're planning on using CPU:
     ```bash
     $ sudo docker-compose build sova-asr
     ```

*	Run the desired service container
     *   GPU (check that you're using GPU in **config.ini** (*cpu* should be set to 0):
     ```bash
     $ sudo docker-compose up -d sova-asr-gpu
     ```
     *   CPU:
     ```bash
     $ sudo docker-compose up -d sova-asr
     ```

## Testing

To test the service you can send a POST request:
```bash
$ curl --request POST 'http://localhost:8888/asr/' --form 'audio_blob=@"Data/test.wav"'
```

## Customizations

You may want to train your own acoustic model, in order to do so go through [PuzzleLib tutorials](https://puzzlelib.org/tutorials/Wav2Letter/). Check [KenLM documentation](https://kheafield.com/code/kenlm/) for building your own language model. This repository was tested on Ubuntu 18.04 and has pre-built .so Trie decoder files for Python 3.6 running inside the Docker container, for modifications you can get your own .so files using [Wav2Letter++](https://github.com/facebookresearch/wav2letter) code for building Python bindings. Otherwise you can use a standard Greedy decoder (set in config.ini).
