from setuptools import find_packages, setup

package_name = "robot_grasping"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="edgar",
    maintainer_email="edgar.welte@kit.edu",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pointcloud_subscriber_node = vmf_contact_main.pointcloud_subscriber_node:main",
        ],
    },
)