---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: python-notes
title: python-notes
description: notes on Python
date: 2024-11-26
image: /thumbnails/backbones.png
---
## Iterables
### Reverse iteration

```python
for i in range(len(nums)-1, -1, -1):
	print nums[i]
```

```python
for i in reversed(range(0,len(nums))):
```

### Flatten a list of lists
There is no builtin "flatten", you got to do it manually like:

```python
python flat_list = [item for sublist in nested_list for item in sublist] 
```

```python
python flat_list = [] 
for sublist in nested_list: 
	flat_list += sublist
```

## Sorting
* `sorted()` function returns a new list containing all items from the iterable in ascending order (by default)
*  **`.sort()`**: Sorts a list **in place**.
## Handy data structures

### Default Dict
```
From collections import defaultdict

d = defaultdict(int) => defaults unknown keys to 0

d = defaultdict(list) => defaults unknown keys to []
```
### Deque
```python
From collections import deque
q = deque()
q.popleft() # Pop from the front
q.append() # Push to the bac
```
### Sets

```python
# Create a set 
my_set = {1, 2, 3} 
# Add / remove an element 
my_set.add(4)
my_set.remove(2)
# Add a bunch of elements from any iterable
my_set.update([5,6,7])
# Check existance
cond = 2 in my_set
# Clear
my_set.clear()
```

## Handy syntax

* variable = val1 if cond else val2

## Graphs

You can encode graphs as hash tables node -> list of neighbors

Weighted graphs can be
node -> list of (neighbor x edge_weight)
## Tree search alghorithms

### BFS
Implement using a queue

e.g.
```python
def bfs(start, target):
	if start not in variable_graph or target not in variable_graph:
		return -1

	queue = deque()
	visited = set()
	
	queue.append([start,1])
	visited.add(start)

	while queue:
		n, w = queue.popleft()
		if n==target:
			return w
	
		for (neighbor, edge) in variable_graph[n]:
			if neighbor not in visited:
				queue.append([neighbor, w*edge])
				visited.add(neighbor)

	return -1
```


## Pointers
BE VERY CAREFUL ABOUT PASSING MUTABLE OBJECTS E.G. LISTS AS ARGUMENTS OF RECURSIVE FUNCTIONS! IT'S ALL PASSED BY ARGUMENT.

### nonlocal: mutable vs imutable

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        
        retv = 0  # Variable to hold the result

        def dfs(node: Optional[TreeNode], acum: int):
            nonlocal retv  # Reference to retv from the outer function
            
            if node is None:
                return
            
            acum = 10 * acum + node.val  # Update acum with the current node's value
            
            if node.left is None and node.right is None:  # Leaf node check
                retv += acum  # Add the accumulated number to retv
            else:
                dfs(node.left, acum)  # Recur on the left child
                dfs(node.right, acum)  # Recur on the right child

        dfs(root, 0)  # Start DFS with the root node and an initial accumulated value of 0

        return retv  # Return the final result
```

If retv was a list variable, you don't need to specify nonlocal.

# Classes

```python
class ClassName:
    """Class docstring describing its purpose."""

    # Class attribute (shared by all instances)
    class_attribute = "value"

    def __init__(self, arg1, arg2):
        """The constructor method initializes instance attributes."""
        self.instance_attribute1 = arg1
        self.instance_attribute2 = arg2

    def method(self):
        """Example of an instance method."""
        return f"Instance attributes: {self.instance_attribute1}, {self.instance_attribute2}"

    @classmethod
    def class_method(cls):
        """Example of a class method."""
        return f"Class attribute: {cls.class_attribute}"

    @staticmethod
    def static_method():
        """Example of a static method."""
        return "This method doesn't depend on instance or class attributes."

```

Notes:
- A **class method** in Python is a method that is bound to the class itself, rather than any specific instance of the class. It operates on class attributes and can be called directly on the class without creating an instance.
- **Class attributes** are variables defined directly within a class, outside any instance method. They are shared by all instances of the class unless explicitly overridden by instance-specific attributes.
	- They can be accessed using the class name (`MyClass.class_attribute`) or through any instance (`instance.class_attribute`).
	- When modified directly on the class, the change is reflected for all instances that rely on the class attribute.  `MyClass.class_attribute = "updated"`
	- If a class attribute is accessed through an instance and modified, Python **creates an instance attribute** with the same name, leaving the original class attribute unchanged. 
	`instance.class_attribute = "modified" # Creates an instance attribute`
	* A class method is a safe way to modify class attributes. Changes made through a class method affect the class attribute and are reflected across all instances.
	* They are conceptually similar to static variables in C+

You can use class attributes for constants, shared data, or type annotations. They are accessible as `LegacySurvey.Target` or `LegacySurvey.Split`.

## Setters, getters and deleters

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value

    @radius.deleter
    def radius(self):
        print("Deleting radius")
        del self._radius

# Creating an instance
c = Circle(5)

# Accessing radius (getter)
print(c.radius)  # 5

# Setting radius (setter)
c.radius = 10
print(c.radius)  # 10

# Attempting to set an invalid radius
try:
    c.radius = -1
except ValueError as e:
    print(e)  # "Radius must be positive"

# Deleting radius (deleter)
del c.radius  # "Deleting radius"
```
# Type hints

## Union
The **`Union`** type hint in Python comes from the `typing` module. It allows you to specify that a variable or function argument can accept multiple types.

```python
from typing import Union

def process(data: Union[int, str]) -> str:
    if isinstance(data, int):
        return f"Integer: {data}"
    elif isinstance(data, str):
        return f"String: {data}"
```

# Misc

* When you see `*` in the argument list, it signifies that **positional arguments are no longer allowed** after the `*`, and all remaining arguments must be explicitly passed using their names. This is commonly done to ensure clarity and to avoid mistakes when calling functions or methods.
* in Python, you can write `45_000_000` instead of `45000000`, and it is **equivalent**.

# Randomness

## Random

You can `import random` and use the builtin random module
- **`random.randint(a, b)`**: Returns a random integer between `a` and `b` (inclusive).
- **`random.random()`**: Returns a random float between `0.0` and `1.0`.
- **`random.choice(sequence)`**: Returns a randomly selected element from the given sequence (like a list).
- **`random.shuffle(sequence)`**: Shuffles the elements of a sequence in place.
- **`random.sample(population, k)`**: Returns a list of `k` unique elements chosen randomly from the population.

## np.random

More advanced random functions like sampling from specific probabilty distributions, etc.

```python
import numpy as np 
np.random.randint(1, 10, size=5) # Random integers between 1 and 10, 5 values 
np.random.normal(0, 1, size=100) # Generate 100 random values from a normal distribution
```