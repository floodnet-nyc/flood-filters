import setuptools

setuptools.setup(
    name='flood_filters',
    version='0.0.1',
    description='Filtering flood measurements from ultrasonic range sensors.',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy'
    ],
    extras_require={})