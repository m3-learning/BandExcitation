# building the wheel
python setup.py sdist bdist_wheel

# install from local build
pip install .

# Upload to pypi
python -m twine upload --repository pypi dist/*

# docs
make clean
make html





