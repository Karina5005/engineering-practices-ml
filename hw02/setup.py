import setuptools

setuptools.setup(name='hw_package_ml',
                 version='1.2',
                 install_requires=['numpy==1.23.4', 'scikit-learn==1.1.3', 'matplotlib==3.6.2', 'opencv-python==4.6.0.66', 'build==0.9.0'],
                 packages=['src']
                 )
