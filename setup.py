import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
    name="tail_extrap",
    version="0.1.0",
    description="Tail extrapolation of a joint probability distribution to "
        "construct environmental contours",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ChiQiao/tail_extrap",
    author="Chi Qiao",
    author_email="qiaochi1990@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["tail_extrap"],
    include_package_data=True,
    install_requires=["scikit-learn"],
)