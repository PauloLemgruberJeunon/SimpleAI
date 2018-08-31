install:
	sudo apt-get -y install python3.6 python3-pip
	python3 -m pip install virtualenv
	virtualenv -p python3 ./venv
	./venv/bin/pip install -r requirements.txt\
