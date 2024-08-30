from setuptools import setup

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
    name='tsp_ip',
    version='0.0.7',
    description='TSP-IP (Travelling Salesman Problem integer programing)',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=['tsp_ip'],
    author='rebui1der',
    author_email='vlad_wizard@mail.ru',
    zip_safe=False,
    url='https://github.com/rebui1der/tsp-ip',
    install_requires=[
        'networkx>=3.3',
        'numpy>=1.26.4',
        'PuLP>=2.7.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11'
    ],
    python_requires='>=3.11'
)
