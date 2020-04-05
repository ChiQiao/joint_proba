from setuptools import setup

setup(
    name='pyEC',
    description='Environmental Contour construction',
    version='0.1',
    url='https://github.com/ChiQiao/pyEC',
    license='MIT',
    author='Chi Qiao',
    author_email='qiaochi1990@gmail.com',
    install_requires=['scikit-learn'],
    packages=['pyEC'],
)

import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
    name="tail-extrapolate",
    version="0.1.0",
    description="Extrapolate the tail of a joint probability distribution",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/realpython/reader",
    author="Real Python",
    author_email="office@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)