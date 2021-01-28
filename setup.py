import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyhesive", # Replace with your own username
    version="0.1.2",
    author="Jacob Faibussowitsch",
    author_email="jacob.fai@gmail.com",
    description="Insert cohesive elements into any mesh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Jfaibussowitsch/pyhesive",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy',
        'scipy',
        'meshio',
        'pymetis'
    ],
    python_requires='>=3.6',
)