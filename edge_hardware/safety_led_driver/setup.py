from setuptools import find_packages, setup

package_name = 'safety_led_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/safety_led_driver.launch.py']),
    ],
    install_requires=['setuptools', 'aiortc'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='The driver for safety LED light with launch support',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'safety_led_driver_node = safety_led_driver.safety_led_driver_node:main',
        ],
    },
)
