from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

packages = ['CloudSentiment', 'RNN_model']
package_dir = {
    'CloudSentiment': 'CloudSentiment',
    'RNN_model': 'RNN_model'}

setup(name='root',
      version="1.0",
      description="Package that does sentiment analysis",
      packages=packages,
      package_dir = package_dir,
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
#    scripts=['scripts/btc_sent_fc-run'],
      zip_safe=False)
