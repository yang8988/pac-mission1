from setuptools import find_packages, setup

package_name = "palletizing_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/default.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "matplotlib", "pyyaml"],
    zip_safe=True,
    maintainer="LYL",
    maintainer_email="kite@changufix.top",
    description="IRAP mixed palletizing planner core (ROS-independent).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "irap_demo = palletizing_core.demo:main",
            "irap_benchmark = palletizing_core.benchmark:main",
        ],
    },
)
