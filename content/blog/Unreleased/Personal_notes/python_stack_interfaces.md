---
tags:
aliases: 
publish: false
slug: python-stack
title: Notes on Numpy, Pandas, etc.
description: Assorted notes and takeaways from discussions
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Python 

- Python2 is deprecated.
- Python3 was released in 2008
- Latest stable version is 3.12 ; industry standard is 3.9 or 3.10.

| **Version** | **Release Date** | **Key Features**                                                                                                 |
|-------------|-------------------|-----------------------------------------------------------------------------------------------------------------|
| **3.8**     | October 2019      | - **Walrus Operator** (`:=`) for inline assignments.                                                           |
|             |                   | - **Positional-Only Arguments**: Improve API clarity.                                                          |
|             |                   | - f-strings support `=` for easier debugging: `f"{var=}"`.                                                     |
|             |                   | - `math.prod()`, `math.isqrt()`, `statistics.fmean()`, and `statistics.geometric_mean()` functions.             |
|             |                   | - `typing.Literal` for stricter type hints.                                                                    |
| **3.9**     | October 2020      | - **Dictionary Union Operators**: `|` and `|=` for merging dictionaries.                                       |
|             |                   | - Simplified type hinting: `list[int]` instead of `List[int]`.                                                 |
|             |                   | - `str.removeprefix()` and `str.removesuffix()` for easier string handling.                                     |
|             |                   | - `zoneinfo` module for timezone support.                                                                      |
| **3.10**    | October 2021      | - **Structural Pattern Matching**: Similar to switch-case (using `match` and `case` statements).               |
|             |                   | - Improved error messages and syntax suggestions.                                                              |
|             |                   | - `Union` type operator (`|`) for type hints (e.g., `int | str`).                                              |
|             |                   | - Parenthesized context managers: `with (open('file') as f, open('file2') as f2):`.                             |
| **3.11**    | October 2022      | - Significant **performance improvements** (10-60% faster execution for most code).                            |
|             |                   | - `Self` type in typing for better class method annotations.                                                   |
|             |                   | - Enhanced `asyncio` with **Task Groups** and **Exception Groups**.                                            |
|             |                   | - `tomllib` module for reading TOML files.                                                                     |
| **3.12**    | October 2023      | - **FrozenDict** in `collections`: Immutable dictionaries.                                                     |
|             |                   | - More informative error messages, especially for `TypeError`.                                                 |
|             |                   | - `typing.override` decorator for explicitly marking overridden methods.                                       |
|             |                   | - `contextvars` improvements for better context management.                                                    |



# Numpy

Numerical Python - Fundamental library for large multi dimensional-arrays and matrices AND functions to operate on those arrays efficiently.

- Replace python lists with arrays for faster computations
- Vectorize operations on entire arrays

# NumPy Cheatsheet

| **Category**               | **Method**                     | **Description**                                                                 |
|----------------------------|---------------------------------|---------------------------------------------------------------------------------|
| **Array Creation**         | `np.array()`                   | Convert lists to arrays.                                                      |
|                            | `np.zeros((n, m))`             | Create a zero-filled array of shape `(n, m)`.                                 |
|                            | `np.ones((n, m))`              | Create a one-filled array of shape `(n, m)`.                                  |
|                            | `np.arange(start, stop, step)` | Create an array with evenly spaced values.                                    |
|                            | `np.linspace(start, stop, n)`  | Create `n` evenly spaced values between start and stop.                       |
|                            | `np.random.rand(n, m)`         | Generate random numbers from a uniform distribution.                         |
|                            | `np.random.randn(n, m)`        | Generate random numbers from a normal distribution.                          |
| **Array Operations**       | `+`, `-`, `*`, `/`, `**`       | Element-wise addition, subtraction, multiplication, division, and exponentiation. |
|                            | `np.add(a, b)`                 | Same as `a + b` (rarely used explicitly).                                     |
|                            | `np.subtract(a, b)`            | Same as `a - b` (rarely used explicitly).                                     |
|                            | `np.multiply(a, b)`            | Same as `a * b` (rarely used explicitly).                                     |
|                            | `np.divide(a, b)`              | Same as `a / b` (rarely used explicitly).                                     |
|                            | `np.exp(a)`                    | Compute element-wise exponential.                                            |
|                            | `np.log(a)`                    | Compute element-wise natural log.                                             |
| **Statistics**             | `np.mean(a)`                   | Compute the mean.                                                             |
|                            | `np.std(a)`                    | Compute the standard deviation.                                               |
|                            | `np.var(a)`                    | Compute the variance.                                                         |
|                            | `np.median(a)`                 | Compute the median.                                                           |
| **Reshaping**              | `a.reshape(n, m)`              | Reshape array to shape `(n, m)`.                                              |
|                            | `a.flatten()`                  | Flatten a multi-dimensional array.                                            |
|                            | `a.T`                          | Transpose the array.                                                          |
| **Indexing**               | `a[1, 2]`                      | Access element at position `(1, 2)`.                                          |
|                            | `a[:, 1]`                      | Access all rows of column 1.                                                  |
|                            | `a[a > 0]`                     | Conditional indexing using a boolean mask.                                    |
| **Linear Algebra**         | `np.dot(a, b)`                 | Compute dot product.                                                          |
|                            | `np.linalg.inv(a)`             | Compute the inverse of a matrix.                                              |
|                            | `np.linalg.eig(a)`             | Compute eigenvalues and eigenvectors.                                         |
| **Missing Data**           | `np.isnan(a)`                  | Identify NaN values.                                                          |
|                            | `np.nanmean(a)`                | Compute the mean, ignoring NaNs.                                              |


An `array` or `ndarray` is the main type defined in numpy.

