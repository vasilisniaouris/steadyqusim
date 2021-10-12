import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="steadyqusim",
    version="0.1.0.0",
    author="Vasilis Niaouris",
    author_email="vasilisniaouris@gmail.com",
    description="Steady state quantum simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vasilisniaouris/steadyqusim",
    packages=setuptools.find_packages(),
    package_dir={'steadyqusim': 'steadyqusim'},
    package_data={'steadyqusim': ['examples/data/*.csv']},
    install_requires=['matplotlib',
                      'numpy',
                      'pandas',
                      'scipy',
                      'qutip'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
