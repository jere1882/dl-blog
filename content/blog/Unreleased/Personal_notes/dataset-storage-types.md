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
| Storage Format                              | Description                                                                                                                                                                                           | File Extension(s)                    |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **HDF5**                                    | Hierarchical data format that supports large datasets, with fast I/O operations and compression. Often used for storing arrays, especially for deep learning models.                                  | `.h5`, `.hdf5`                       |
| **Hugging Face Datasets (Arrow)**           | A format used by Hugging Face's datasets library. Built on Apache Arrow, designed for efficient storage and fast access to data, especially for large-scale NLP datasets.                             | `.arrow`                             |
| **Simple Folders with Plain Files**         | A straightforward method where each file (e.g., images or text) is stored in directories. Often used for unstructured data. Files are stored as they are, without additional metadata or compression. | `.jpg`, `.png`, `.txt`, `.csv`, etc. |
| **CSV (Comma Separated Values)**            | A simple text-based format for storing tabular data where each line represents a row and columns are separated by commas. Widely used for smaller datasets.                                           | `.csv`                               |
| **Pickle**                                  | A Python-specific binary format for serializing Python objects. Can store more complex data structures, like lists or dictionaries, but not ideal for large-scale datasets.                           | `.pkl`, `.pickle`                    |
| **TFRecord (TensorFlow)**                   | A format used by TensorFlow for storing large datasets, optimized for parallel reading and data pipeline processing. Useful for large-scale training tasks.                                           | `.tfrecord`, `.tfrecords`            |
| **JSON**                                    | A flexible text-based format that supports hierarchical data structures, commonly used for storing metadata or datasets with nested information.                                                      | `.json`                              |
| **Parquet**                                 | A columnar storage format optimized for big data processing. Efficient for storing large datasets with structured data, offering good compression and fast query performance.                         | `.parquet`                           |
| **LMDB (Lightning Memory-Mapped Database)** | A fast, memory-mapped key-value store used for storing datasets, especially in deep learning tasks. It is efficient for large-scale datasets and supports fast read/write operations.                 | `.lmdb`                              |
| **SQLite**                                  | A relational database format that can store tabular data in a single file. Useful for small- to medium-sized datasets with complex relationships.                                                     | `.sqlite`, `.db`                     |
| **Avro**                                    | A binary format used for serialization, often in big data environments. Supports schema evolution and is optimized for high-volume storage.                                                           | `.avro`                              |
| **Feather**                                 | A fast, lightweight binary columnar data format designed to store data frames with efficient read/write operations. Often used for fast, in-memory data sharing between Python and R.                 | `.feather`                           |
| **Excel (XLS/XLSX)**                        | A spreadsheet format widely used for business applications. Can store tabular data and provide a graphical interface for users.                                                                       | `.xls`, `.xlsx`                      |
# Up and downsides

| Storage Format       | Advantages                                                                                         | Disadvantages                                                                                   |
|----------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **HDF5**             | - Efficient for large datasets.   - Supports compression and fast I/O.   - Good for deep learning and scientific computing. | - Not human-readable.   - Requires external libraries for use (e.g., `h5py`).               |
| **Hugging Face Datasets (Arrow)** | - Optimized for fast I/O and distributed processing.   - Easy integration with Hugging Face ecosystem.   - Efficient storage with compression. | - Less common outside of Hugging Face and NLP tasks.   - Requires Arrow library.             |
| **Simple Folders with Plain Files** | - Easy to implement.   - File formats are widely compatible (e.g., images, text).   - Suitable for unstructured data. | - Slow access for large datasets.   - No inherent compression.   - No metadata support.   |
| **CSV**              | - Simple, widely used format.   - Human-readable.   - Supported by many tools and libraries. | - Inefficient for large datasets.   - Lack of support for complex data types.   - No compression. |
| **Pickle**           | - Efficient for serializing Python objects.   - Stores complex data structures.   - Fast read/write. | - Not human-readable.   - Python-specific (not cross-language).   - Can be insecure if untrusted data is loaded. |
| **TFRecord (TensorFlow)** | - Optimized for TensorFlow models and big data pipelines.   - Efficient read/write performance. | - Primarily designed for TensorFlow, limiting cross-tool compatibility.   - Not human-readable. |
| **JSON**             | - Human-readable.   - Supports hierarchical data.   - Widely used for web and API data. | - Less efficient for large datasets.   - Can become hard to manage for very large, complex datasets. |
| **Parquet**          | - Efficient storage and fast querying for large datasets.   - Supports compression.   - Optimized for big data tools. | - Not human-readable.   - Requires additional libraries (e.g., PyArrow) for use.            |
| **LMDB (Lightning Memory-Mapped Database)** | - Fast and efficient for read-heavy workloads.   - Memory-mapped storage improves performance.   - Supports complex datasets. | - Not as widely supported as other formats.   - Not ideal for write-heavy operations.         |
| **SQLite**           | - Lightweight relational database.   - Supports SQL queries.   - Single file for entire dataset. | - Slower for large datasets compared to more specialized formats.   - Not optimized for big data workflows. |
| **Avro**             | - Optimized for high-volume data storage.   - Supports schema evolution.   - Compact binary format. | - Not human-readable.   - Can require additional setup for reading and writing.             |
| **Feather**          | - Very fast read/write operations.   - Ideal for in-memory data sharing between Python and R.   - Lightweight. | - Not as widely supported as CSV or Parquet.   - Not suitable for large-scale, distributed datasets. |
| **Excel (XLS/XLSX)** | - Easy to use with graphical interfaces.   - Well-supported in business environments.   - Good for small datasets and quick data analysis. | - Slow for large datasets.   - No support for complex data types (compared to databases).   - Limited for machine learning workflows. |
