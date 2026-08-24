import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'spot_terrain_gridmap'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@example.com',
    description=(
        'Pipeline immagine -> gridmap del terreno (DINOv2 + proiezione BEV) '
        'per navigazione energy-aware del robot quadrupede Spot.'
    ),
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'terrain_feature_node = spot_terrain_gridmap.terrain_feature_node:main',
            'feature_extraction = spot_terrain_gridmap.feature_extraction_node:main',
        ],
    },
)
