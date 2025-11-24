from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This funciton will return list of requiremets"""
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="brain-tumor-classifier",
    version="0.0.1",
    author="Parag",
    author_email="paraggupta1976@gmai.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)