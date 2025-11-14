from setuptools import find_packages, setup

setup(
    name='analisis_riesgo_crediticio',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Analisis de riesgo crediticio del equipo 14 de la materia de MLOPS Fase 2',
    author='mlops eq14',
    license='MIT',
    python_requires='>=3.8',
)
