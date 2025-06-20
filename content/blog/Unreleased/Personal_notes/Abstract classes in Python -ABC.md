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
# ABC

The ABC class in Python is a helper class that provides a standard way to create ABSTRACT BASE CLASSES using inheritance. 

The ABC class uses ABCMeta as its metaclass. Any subclass of ABC cannot be instantiated unless all of its abstract methods are overriden.

The primary purpose of the ABC class is to serve as a base class for other classes that want to define abstract methods. By inheriting from ABC, a class can define methods as abstract using the @abstractmethod decorator. This enforces that any subclass must implement these methods.

```python
from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    @abstractmethod
    def my_abstract_method(self):
        pass

class ConcreteClass(MyAbstractClass):
    def my_abstract_method(self):
        print("Implemented abstract method")

# This will work
concrete_instance = ConcreteClass()
```

Python doesn't have a built-in syntax for defining abstract classes and methods like C++ does with virtual ... = 0. Instead, Python uses the abc module to provide this functionality.

Example:


```python
class ImageDenoiser(ABC):
    @abstractmethod
    def denoise(self, image: Union[str, Image.Image, np.ndarray], save_path: Optional[str] = None) -> Image.Image:
        """
        Denoise an image.
        
        Args:
            image: Either a path to an image or a PIL Image/numpy array.
            save_path (str, optional): Path to save the denoised image.
            
        Returns:
            PIL.Image: Denoised image.
        """
        pass
```