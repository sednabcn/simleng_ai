import setuptools
import versioneer
from setuptools import find_packages, setup
from simleng_ai import simleng


def readme():
    with open("Readme.md") as f:
        long_description = f.read()
        return long_description


setup(
    name="simleng_ai",  # unknown option
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A module to develop Data_Analysis starategies",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # License, see https://choosealicense.com
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url=[
        "https://github.com/sednabcn/simleng_ai",
        "https://github.com/sednabcn/simleng_ai/issues",
    ],
    entry_points={
        "console_scripts": [""],
    },
    author="Ruperto Pedro Bonet Chaple",
    author_email="ruperto.bonet@modelphysmat.com",
    license="MIT",
    # packages=["simleng_ai"],
    readme="Readme.md",  # unknown option
    python_requires=">=3.8",
    keywords="simleng data_analisys setuptools development",
    # manifest_in=["./input_file/simleng.txt","./datasets/*"],
    install_requires=[
        "biokit==0.5.0",
        "colormap==1.0.4",
        "matplotlib==3.7.1",
        "numpy==1.24.4",
        "packaging==21.3",
        "pandas==2.0.3",
        "scipy==1.8.0",
        "statsmodels==0.14.0",
        "scikit_learn==1.3.0",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    scripts=["bin/api.py"],
    include_package_data=True,
    zip_safe=False,
)
