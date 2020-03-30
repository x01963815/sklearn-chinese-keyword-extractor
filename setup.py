from setuptools import setup, find_packages

setup(
    name='chinese_keyword_extractor',
    version='0.1.0',
    description='Extract keywords from Chinese product name.',
    py_modules=['text'],
    author='Ferris Liu',
    author_email='x01963815@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy>=1.16.4'
                      'scikit-learn>=0.19.2'],
    python_requires='>=2.7',
)