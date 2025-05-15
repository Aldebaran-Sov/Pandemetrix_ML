FROM python:3.9-slim

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    zsh \
    curl \
    wget \
    fonts-powerline \
    && rm -rf /var/lib/apt/lists/*

# Configuration de Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Définition de zsh comme shell par défaut
RUN chsh -s $(which zsh)
ENV SHELL=/bin/zsh

WORKDIR /Pandemetrix_ML

# Copy the requirements file
COPY ./requirements.txt /Pandemetrix_ML/

# Update pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
# COPY . /Pandemetrix_ML

# Commande par défaut pour démarrer zsh au lieu de bash
CMD ["zsh"]
