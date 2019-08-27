---
title: "Dealing with Grid Data in Python"
date: "2016-12-08"
template: "post"
slug: "/posts/dealing-with-grid-data-in-python"
category: "Data Science"
tags: 
  - "python"
  - "numpy"
description: "In my PhD research, I do a lot of analysis of 2D and 3D grid data
output by simulations I run. In my analyses, it's very helpful to restructure
these data into a more useable format. A few key lines of python code do the
trick."
---

In my PhD research, I do a lot of analysis of 2D and 3D grid data output by
simulations I run. If, for example, I had a toy system that was 3x2x2 grid
points, the raw data would be structured sort of like this:

```
x   y   z   value
0   0   0   0.9
1   0   0   1.1
2   0   0   0.8
0   1   0   1.1
1   1   0   1.0
2   1   0   0.9
0   0   1   0.6
1   0   1   1.2
2   0   1   0.8
0   1   1   0.9
1   1   1   1.2
2   1   1   1.3
```

In my analyses, it's very helpful to restructure these data into a format
where, in this case, `x = [0, 1, 2]`, `y = [0, 1]`, `z = [0, 1]`, and `value`
is a 3D array such that `value[i, j, k]` returns the value corresponding to
position `(x[i], y[j], z[k])`.

It's easy to do that in just a few lines. Say the above raw data is stored in
`data.dat`.

```python
>>> import numpy as np
>>> x, y, z, value = np.loadtxt('data.dat', skiprows=1).T
>>> x, y, z = np.unique(x), np.unique(y), np.unique(z)
>>> nx, ny, nz = len(x), len(y), len(z)
>>> value = value.reshape((nz, ny, nx)).T
```

Note that if the raw data had `x` varying the slowest and `z` varying the
fastest, the final line would look like `value = value.reshape((nx, ny,
nz))`.

Finally, if you want to go the other way, where you have your `x`, `y`, and
`z` arrays and 3D `values` array, you can make use of the
`sklearn.utils.extmath.cartesian` function (first introduced on this
[StackOverflow post](http://stackoverflow.com/a/1235363/2680824). If you want
`z` to be the fastest changing variable, it would look something like this:

```python
>>> from sklearn.utils.extmath import cartesian
>>> import numpy as np
>>> x, y, z = [0, 1, 2], [0, 1], [0, 1]
>>> nx, ny, nz = len(x), len(y), len(z)
>>> value = np.arange(nx*ny*nz).reshape((nx,ny,nz)) # define 3D value array
>>> xyz = cartesian((z, y, x))
>>> value = value.flatten()
>>> np.hstack((xyz, value[:,None]))
array([[ 0,  0,  0,  0],
        [ 0,  0,  1,  1],
        [ 0,  1,  0,  2],
        [ 0,  1,  1,  3],
        [ 1,  0,  0,  4],
        [ 1,  0,  1,  5],
        [ 1,  1,  0,  6],
        [ 1,  1,  1,  7],
        [ 2,  0,  0,  8],
        [ 2,  0,  1,  9],
        [ 2,  1,  0, 10],
        [ 2,  1,  1, 11]])
```

The `value[:,None]` thing on the last line adds an extra dimension to the 1D
value array so the elements of the tuple passed to `np.hstack` are both 2D
numpy arrays.

```python
>>> value.shape
(12,)
>>> value[:,None].shape
(12, 1)
>>> value[None,:].shape
(1, 12)
```
