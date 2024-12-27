---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: python-package-strucure-notes
title: python-notes
description: notes on Python
date: 2024-11-26
image: /thumbnails/backbones.png
---
- A **module** is a single Python file (with the `.py` extension) that contains Python code, such as functions, classes, and variables, as well as runnable code.
-  You can think of it as a single unit of code that can be imported into other Python programs using the `import` statement.

```python
# mymodule.py
def greet():
    print("Hello, world!")
```

```python
import mymodule
mymodule.greet()  # Output: Hello, world!
```

- A **package** is a directory that contains multiple Python modules, along with an `__init__.py` file, which tells Python that the directory should be treated as a package.
- A package allows you to structure larger applications by grouping related modules together in a directory hierarchy such as:

```
my_package/
    __init__.py
    module1.py
    module2.py
```

modules can then be imported from the package as follows:

```
from my_package import module1
from my_package.module2 import some_function
```

Example:

``` bash
checkbox_detector/
├── checkbox_detector/
│   ├── __init__.py            # Initialization of the checkbox_detector package
│   ├── preprocessing.py       # Image preprocessing utilities (e.g., thresholding)
│   ├── detection.py           # Functions for detecting checkboxes in images
│   ├── classification.py      # (Future functionality) Classify detected checkboxes
│   └── utils.py               # Utility functions (e.g., bounding box intersection, overlap checking)
├── tests/
│   ├── __init__.py            # Initialization of the tests package
│   └── test_checkbox_detector.py  # Unit tests for the package
│   └── test_data              # Simple crops for unit testing
├── scripts/
│   └── run_detector.py        # Example script for detecting checkboxes using the package
├── setup.py                   # Installation script for the package
├── data/images                # Testing images
└── README.md                  # This file
```