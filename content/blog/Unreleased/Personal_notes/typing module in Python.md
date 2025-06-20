---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: typing-in-python
title: typing-in-python
description: notes on Python
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Overview of the `typing` Library in Python

The `typing` library in Python is **official**. It was introduced in **Python 3.5** (PEP 484) and has been continuously expanded in later versions. It provides support for **type hints**, enabling static type checking and better code clarity.

## **Key Features of `typing`**

### **1. Basic Type Hints**
- `int`, `float`, `str`, `bool`, `None`
- Example:

```python
  def add(x: int, y: int) -> int:
      return x + y
```

### **2. Generics and Collections**

- `List[T]`, `Tuple[T, ...]`, `Dict[K, V]`, `Set[T]`
- Example:
```python
 from typing import List, Dict  
 def get_scores(names: List[str]) -> Dict[str, int]:     
	 return {name: len(name) for name in names}
```
### **3. Union and Optional Types**
- `Union[A, B]`: Variable can be of type A or B.
- `Optional[T]`: Equivalent to `Union[T, None]`

```python
from typing import Union, Optional  
def process(value: Union[int, float]) -> float:     
	return float(value)  
	
def maybe_return(n: int) -> Optional[str]:     
	return str(n) if n > 0 else None
```

### **4. Callable**
Used for function signatures.

```python
from typing import Callable  

def apply_func(f: Callable[[int, int], int], x: int, y: int) -> int:
	return f(x, y)
```

### **5. Type Aliases**

Assign meaningful names to complex types.
```python
from typing import Dict, List  
UserDB = Dict[str, List[str]]  
def get_friends(user_db: UserDB, user: str) -> List[str]:     
	return user_db.get(user, [])
```

### **6. New Features in Python 3.9+**

Built-in generics (`list`, `dict`, `tuple`) can replace `List`, `Dict`, etc.
```python
def greet(names: list[str]) -> dict[str, int]:     
	return {name: len(name) for name in names}
```
### **7. New Features in Python 3.10+**

`Type | Type` instead of `Union[Type, Type]`

```python
def process(value: int | float) -> float:     
	return float(value)
```
