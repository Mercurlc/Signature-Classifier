from setuptools import setup, find_packages

setup(
    name='Signature-Classifier',
    version='1.0.0',
    author='Mercurlc',
    author_email='alexanderarefievrus@gmail.com',
    description='A package for signature classification with HOG features and KNN classifier, also for forg signatures detection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Mercurlc/Signature-Classifier',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'optuna',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'seaborn',
        'sphinx'
    ],
    project_urls={
        'Documentation': 'docs',
        'Example': 'example-usage'
      },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
