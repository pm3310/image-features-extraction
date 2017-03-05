## Image Features Extraction

This is the Keras model of the 16-layer convolutional network used by the VGG team in the ILSVRC-2014 competition.
It can be used in order to extract features from a visual point of view. The features can be utilized for
many applications such as indexing of images or image recommendations.

More specifically, the values of the penultimate layer of the CNN can be used as features for an image.

### Python Installation
1. Download Python 3.5.2 https://www.python.org/downloads or install it via Homebrew
2. Put to your bash profile `export PATH="/Library/Frameworks/Python.framework/Versions/3.5/bin:${PATH}"`
 so that virtualenv is connected to Python 3.5.2
3. Install pip3.5 if is not installed already
4. Then, install virtualenv for Python 3.5: `pip3.5 install virtualenv`

### Install using requirements.txt
1. cd into image-features-extraction
2. run command: `virtualenv image-features-extraction-venv` (this should be done only once)
3. run command: `source image-features-extraction-venv/bin/activate`
4. run command: `pip3.5 install -r requirements.txt` (in order to install dependencies in the current virtualenv)

### Install using setup.py
1. `python setup.py sdist`

### Example
1. Go to `example/example.py` 

