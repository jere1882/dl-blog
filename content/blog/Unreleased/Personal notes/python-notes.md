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
