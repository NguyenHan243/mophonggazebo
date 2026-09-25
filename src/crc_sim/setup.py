import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'crc_sim'

# Junk created by Windows / Google Drive - never packaged
SKIP = {'desktop.ini', 'Thumbs.db', '.DS_Store'}


def keep(paths):
    return [p for p in paths if os.path.basename(p) not in SKIP]


def data_tree(src_dir):
    """Install a whole directory tree (preserving structure) into share/<pkg>/."""
    out = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        files = keep(files)
        if not files:
            continue
        dest = 'share/' + package_name + '/' + root.replace(os.sep, '/')
        out.append((dest, [os.path.join(root, f).replace(os.sep, '/')
                           for f in files]))
    return out


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', keep(glob('launch/*.launch.py'))),
        ('share/' + package_name + '/urdf', keep(glob('urdf/*.xacro'))),
        ('share/' + package_name + '/worlds', keep(glob('worlds/*'))),
        ('share/' + package_name + '/rviz', keep(glob('rviz/*.rviz'))),
        ('share/' + package_name + '/config', keep(glob('config/*.yaml'))),
    ] + data_tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UEH CRC Team',
    maintainer_email='3itech@ueh.edu.vn',
    description='Gazebo simulation for UEH Creative Robot Contest 2026',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'starter      = crc_sim.starter_node:main',
            'sensor_check = crc_sim.sensor_check:main',
            'snapshot     = crc_sim.snapshot:main',
            'traffic_light = crc_sim.traffic_light_node:main',
            'pedestrian   = crc_sim.pedestrian_node:main',
            'bocua = crc_sim.bocua:main',
            'leocau = crc_sim.leocau:main',
        ],
    },
)
