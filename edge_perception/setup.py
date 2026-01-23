from glob import glob
from setuptools import find_packages, setup

package_name = 'edge_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob('models/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='Semantic segmentation projection onto point clouds for Jetson.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sam_instance_segment_node = edge_perception.segmentation.instance_segment.sam_instance_seg_node:main',
            'yolo_instance_segment_node = edge_perception.segmentation.instance_segment.yolo_instance_seg_node:main',
            'semantic_segment_node = edge_perception.semantic_segment_node:main',
            'instance_segment_node = edge_perception.instance_segment_node:main',
        ],
    },
)
