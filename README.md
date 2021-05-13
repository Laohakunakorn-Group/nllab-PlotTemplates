# nllab-PlotTemplates
Plotting templates in Python and Julia. A gallery of basic plots and their code for you to adapt and use.

## Installation: local
To run locally, make sure you have Python 3 and pipenv installed. Clone and navigate to the repository, then run (for the first time)

	pipenv install
	pipenv shell 

After the first time, you only need to run ```pipenv shell``` to activate the environment, and ```exit``` to leave it. 

Then run ```jupyter notebook``` to launch.

More info on pipenv [here](https://pipenv-fork.readthedocs.io/en/latest/basics.html).

## Installation: Docker
You can run the code from one of our lab's predefined Docker containers. We currently have three Docker containers:

	nadanai263/nllab-python:003
	nadanai263/nllab-jupyter:005
	nadanai263/nllab-julia:004

The numeric tags will increase as the containers are updated, so please check them on [DockerHub](https://hub.docker.com/). To use these, clone the repository, open up a terminal and navigate to the directory containing the repository. Then (making sure you have a working [Docker](https://www.docker.com) installation in place), run (on Mac, Linux)

	docker run -it --rm -v "$PWD":/app nadanai263/nllab-python:003 /bin/bash

On Windows, replace `"$PWD"` with `"%CD%"` (for command prompt) or `${pwd}` (for powershell).

This will pull and launch a Docker container, and start up a Linux shell. Your current working folder will be mounted to `/app`. Run the scripts as required.

#### Running Jupyter notebook in Docker container

You can directly start a Jupyter notebook in your current directory, using

	docker run -p 8888:8888 --rm -it -v "$PWD":/home/jovyan nadanai263/nllab-jupyter:005

Again, on Windows replace `"$PWD"` with `"%CD%"` (for command prompt) or `${pwd}` (for powershell).