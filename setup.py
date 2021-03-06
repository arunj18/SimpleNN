from setuptools import setup

setup(
    name='SimpleNN',
    version='0.0.3',
    description='Simple Multi Layer Neural Network',
    url='git@github.com:arunj18/SimpleNN.git',
    author='Arun John',
    author_email='arunjoh@gmail.com',
    license='MIT',
    packages=['SimpleNN'],
    install_requires = [
        'numpy>=1.19.4'
    ],
    zip_safe=False
)
