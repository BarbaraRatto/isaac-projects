from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'spot_rl_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'),
            glob('models/.gitkeep')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='barbara',
    maintainer_email='barbara@student.it',
    description='RL energy-aware local controller for Spot quadruped',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Nodo: comprime gridmap DINOv2 (384 layer) -> tensor (16+1 layer)
            'energy_costmap_node = spot_rl_controller.energy_costmap_node:main',
            # Nodo: inference RL (carica modello SB3 e pubblica /cmd_vel)
            'rl_controller_node = spot_rl_controller.rl_controller_node:main',
            # Nodo: inference specifico per modello Isaac Lab vettorizzato (Piatto + Fake DINO)
            'rl_isaaclab_node = spot_rl_controller.rl_isaaclab_node:main',
            # Script: training PPO (eseguito standalone, non come nodo ROS2)
            'train_rl = spot_rl_controller.train_rl:main',
        ],
    },
)
