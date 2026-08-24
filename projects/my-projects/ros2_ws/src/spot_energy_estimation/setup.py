import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'spot_energy_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Barbara',
    maintainer_email='you@example.com',
    description='Nodo di stima del consumo energetico per il robot quadrupede, a partire da /joint_states',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'energy_estimation_node = spot_energy_estimation.energy_estimation_node:main',
        ],
    },
)
