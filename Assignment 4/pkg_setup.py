from setuptools import find_packages, setup

# Define the package name consistently
package_name = 'potential_field_planner'

setup(
    name=package_name,
    version='0.0.0',
    # Automatically find all Python packages (directories with __init__.py)
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install resources for ROS 2 package indexing
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install the package.xml file
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ak',
    maintainer_email='adnanabbask@gmail.com',
    description='Potential field-based path planning node for Robile in Gazebo.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # --- CRITICAL SECTION: Define the executable script ---
    entry_points={
        'console_scripts': [
            # EXECUTABLE_NAME = PACKAGE_FOLDER.PYTHON_FILE_NAME:ENTRY_FUNCTION
            # 'planner' is the name you type with 'ros2 run potential_field_planner planner'
            # 'potential_field_planner.potential_field_planner' refers to 
            # the file 'potential_field_planner/potential_field_planner.py'
            'planner = potential_field_planner.robile_path_planning:main'
        ],
    },
)
