from setuptools import find_packages, setup

package_name = 'led_safety_monitor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/safety_monitor.launch.py']),
    ],
    install_requires=['setuptools', 'aiortc'],
    zip_safe=True,
    maintainer='edgebot',
    maintainer_email='yihao.cai@wayne.edu',
    description='Safety monitor node with launch support',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'safety_monitor_node = led_safety_monitor.safety_monitor_node:main',
        ],
    },
)
