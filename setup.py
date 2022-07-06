from setuptools import find_packages, setup
import versioneer


requirements = [
    'psiexperiment[ni,zarr]',
]


setup(
    name='psilbhb',
    author='LBHB development team',
    author_email='lbhb@alum.mit.edu',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    description='Module for running LBHB experiments',
    entry_points={
        'console_scripts': [
            #'psilbhb=psi.application.launcher:main',
        ]
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
