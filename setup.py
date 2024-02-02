from setuptools import find_packages , setup
from typing import List 

hypen_e_dot ='-e.'  


def get_packages(file_path:str)->List[str]:
    with open(file_path) as file_obj :
        requirements= file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements ] 
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
        return requirements

setup(
    name = "House-Price-Prediction",
    version = "1.0.0",
    author = "Siddharth Dhole",
    author_email='shidhudhole358@gmail.com',
    packages = find_packages(),
    install_requires =get_packages('requirements.txt')
    )

