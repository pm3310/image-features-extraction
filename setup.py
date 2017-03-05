from distutils.core import setup

setup(name='image-feature-extraction',
      version='0.1',
      description='Keras model of the 16-layer convolutional network used by the VGG team in order to extract features from images',
      author='Pavlos Mitsoulis-Ntompos (github username: pm3310)',
      author_email='p.mitsoulis@gmail.com',
      packages=['image_net'],
      install_requires=[
          'Keras==1.0.8',
          'numpy==1.11.1',
          'Pillow==3.3.1',
          'requests==2.13.0',
          'scikit-learn==0.17.1',
          'scipy==0.18.0',
          'six==1.10.0',
          'Theano==0.8.2'
      ],
      )
