FROM continuumio/miniconda3
ARG ENV_NAME_CONDA=mieml

# Create conda environment and updating it using environment.yml file from repository
ADD environment.yml /tmp/environment.yml
RUN conda create -n $ENV_NAME_CONDA
RUN conda env update --name $ENV_NAME_CONDA --file /tmp/environment.yml --prune

# Automatically start created conda environment whenever container is launched
RUN echo "source activate $ENV_NAME_CONDA" > ~/.bashrc
ENV PATH /opt/conda/envs/$ENV_NAME_CONDA/bin:$PATH

# Download and extract RVL-CDIP dataset into the relevant directory
RUN apt-get -y update && apt-get -y install curl
RUN git clone https://github.com/RayedSuhail/mie-document-determination.git
# RUN curl -s  https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz | tar xvz - -C mie-document-determination/dataset/images


ADD https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz /rvl-cdip.tar.gz