# Pure Python array (pure_array)

A pure python implementation of the `array` data structure of [numpy](https://numpy.org/). This work is not intended to
be a replacement for numpy, but rather a possible alternative when numpy is not allowed. This implementation has the 
same syntax as numpy, but not all functions are implemented. 

## Installation

```bash
pip install pure_array
```

You can also install the development version from GitHub:

```bash
git clone https://github.com/ferranSanchezLlado/pure_array.git
cd pure_array
pip install -e .
```

## Usage

```python
import pure_array as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

assert a + b == np.array([5, 7, 9])
assert a * b == np.array([4, 10, 18])   
assert a.max() == 3
```

## Requirements

As can be seen from the `requirements.txt` there are no requirements for the package itself, but the tests 
(`requirements_dev.txt`) require [pytest](https://docs.pytest.org/en/stable/) and
[flake8](https://flake8.pycqa.org/en/latest/) can be used for linting.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Future work

- [ ] Implement boolean indexing
- [ ] Documentation
- [ ] Better static typing
- [ ] More tests
- [ ] More functions
