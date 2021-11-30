from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]


setup(
    name='Main_package',
    version="1.0",
    description="Package that does sentiment analysis",
    package_dir={
        'Main_package': 'Main_package',
        'Main_package.CloudSentiment': 'Main_package/CloudSentiment',
        'Main_package.RNN_model': 'Main_package/RNN_model'
    },
    packages=[
        'Main_package', 'Main_package.CloudSentiment', 'Main_package.RNN_model'
    ],
    install_requires=requirements,
    test_suite='tests',
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    #    scripts=['scripts/btc_sent_fc-run'],
    zip_safe=False)
