from setuptools import setup, find_packages
from typing import List

DASH_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    - This function takes file path as a string input & returns the list of packages in that file
    - The file we want is requirements.txt
    - This function reads the file from the specified file path & populates all the mentioned libraries in the file as a list of strings
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = [r.replace('\n', '') for r in file_obj.readlines()]

        if DASH_E_DOT in requirements:
            requirements.remove(DASH_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='V0.0.1',
    author='Abhirup Mukherjee',
    author_email='abhirupmukherjee.fiem@gmail.com',
    maintainer='Abhirup Mukherjee',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
