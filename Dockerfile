#
# Ce dockerfile est destiné à être utilisé avec l'intégration continue
# Il créé l'image à partir du répertoire de travail de Jenkins
#

# Image pour le build de l'environnement
# Base nvidia/cuda:11.0-base
FROM nvidia/cuda:11.4.0-base-ubuntu20.04 AS build

# On configure bash comme étant le shell 
SHELL [ "/bin/bash", "--login", "-c" ]

# On bloque le mode interactif pour pas qu'on nous demande la timezone lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# On installe les paquets nécessaires
RUN apt-get update -yq \
&& apt-get install curl -yq \
&& apt-get install git -yq \
&& apt-get install wget -yq \
&& apt-get install tzdata -yq \
&& apt-get install build-essential gdal-bin libgdal-dev -yq \
&& apt-get clean -y

# On configure la timezone
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# On installe Conda
WORKDIR /var/data/conda/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /tmp/install-miniconda.sh
RUN chmod 775 /tmp/install-miniconda.sh
RUN /tmp/install-miniconda.sh -b -u -p /var/data/conda
RUN echo 'export PATH=/var/data/conda/bin:$PATH' >> /etc/profile
RUN echo 'export PATH=/var/data/conda/bin:$PATH' >> /etc/bashrc
RUN chmod u+x /var/data/conda/bin

# On installe le code OCSGE
WORKDIR /var/data/ocsge/
COPY . /var/data/ocsge/

# On installe Mamba
RUN /var/data/conda/bin/conda install mamba -c conda-forge

# On télécharge les librairies python dans l'environnement Conda
RUN /var/data/conda/bin/mamba env create --file=environment.yml

# On fait en sorte que les commandes RUN soient dans l'environnement conda ocsge
SHELL ["/var/data/conda/bin/mamba", "run", "-n", "ocsge", "/bin/bash", "-c"]

# On package le code, génère commandes exécutables avec nos scripts Python
RUN pip install .

# On installe conda-pack
RUN mamba install -c conda-forge conda-pack

# On utilise conda-pack pour créer un environement "standalone" dans /venv
RUN conda-pack -n ocsge -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar
  
# Cleanup prefixes from in the active environment
RUN /venv/bin/conda-unpack






# Image pour l'executable
# On repart d'une image Linux neuve
FROM nvidia/cuda:11.4.0-base-ubuntu20.04 AS run

# Paramètre de build, le compte utilisateur (à la place de root), si pas de build-arg n'est passé ce sera 'svc_ocsge' par défaut 
ARG DOCKER_USER=svc_ocsge
# UID correspondant
ARG DOCKER_USER_UID=101  
ARG DOCKER_USER_GID=102  

# On bloque le mode interactif pour pas qu'on nous demande la timezone lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# On installe les paquets nécessaires
RUN apt-get update -yq \
&& apt-get install tzdata -yq \
&& apt-get install gdal-bin -yq \
&& apt-get clean -y

# On configure la timezone
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# On créé dans l'image le groupe et le user
RUN adduser --system --group $DOCKER_USER
# Et on lui donne l'UID qui va bien pour que ça corresponde
RUN usermod -u $DOCKER_USER_UID $DOCKER_USER
RUN groupmod -g $DOCKER_USER_GID $DOCKER_USER

# Et on recopie l'environnement Python du build
COPY --chown=$DOCKER_USER:$DOCKER_USER  --from=build /venv /venv

# On installe le code OCSGE
WORKDIR /var/data/ocsge/
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=build /var/data/ocsge/ /var/data/ocsge/


# On copie un script shell de démarrage dans l'image
COPY --chown=$DOCKER_USER:$DOCKER_USER entrypoint.sh /usr/local/bin/
RUN chmod u+x /usr/local/bin/entrypoint.sh

# On dit à Docker d'utiliser cet utilisateur pour toutes les commandes suivantes
USER $DOCKER_USER

# On défini bash comme shell par défaut
SHELL ["/bin/bash", "-c"]

# Le script à lancer comme point d'entrée
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# La commande de départ
CMD bash
