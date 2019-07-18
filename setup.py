from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'matplotlib >= 3',
    'tensor2tensor==1.13.2',
    'matplotlib-scalebar',
    'google-auth-oauthlib',
    'tensorflow-probability==0.6.0',
    'tflearn'
]

setup(
    name='trainer',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
