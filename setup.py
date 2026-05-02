from setuptools import find_packages, setup

setup(
    name='medical_insurance_cost',
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'Flask', 'xgboost', 'matplotlib', 'seaborn']
)