from setuptools import find_packages,setup #automatically find out all the packages in the ml directory created 
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
        This function will return the requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","")for req in requirements.txt] # Replacing \n with a blank, moving in the next line will cause this 

        # We have added '-e .' to connect the requirements file to setup.py 
        # We are just removing the '-e .'
        if HYPEN_E_DOT in requirements:    
            requirements.remove(HYPEN_E_DOT)
    return requirements 

setup(
    name = 'mlproject'
    , version = '0.0.1'
    , author = 'Mihir Vij'
    , author_email = 'vijmihiir@gmail.com'
    , packages = find_packages() # Will use the module imported earlier 
    , install_requires = get_requirements('requirements.txt')
)
