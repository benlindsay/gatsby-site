---
title: "Profiling Jupyter Notebook Code with py-spy"
date: "2021-05-14"
template: "post"
slug: "/posts/profiling-jupyter-notebook-code-with-py-spy"
category: "Data Science"
tags:
  - "jupyter"
  - "py-spy"
  - "performance"
  - "python"
description: "When you sync your Jupyter notebooks with Jupytext, you get to keep all the benefits of Jupyter notebooks while also being able to pass your code through a profiler like py-spy to get a rich, interactive visualization that helps you quickly understand where the bottlenecks are."
---

Jupyter notebooks are great for interactive development and visualization, but it's hard to get nice code profiling visualizations.
Jake VanderPlas has [a great exceprt in his Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html) with examples of how to time and profile code in a Jupyter notebook.
These work great, but the text output isn't always as intuitive as visualizations like [flame graphs](http://www.brendangregg.com/flamegraphs.html).
They give you a hierarchical view of which lines of code at all depths of the call stack are contributing the most to the compute time.
Not only that, they're *interactive* SVGs!
[Click on the image in the flame graphs link](http://www.brendangregg.com/FlameGraphs/cpu-mysql-updated.svg) and you can check out the interactivity yourself.
You can zoom into specific parts of the call stack and search for terms, and hover over blocks to see the full description at the top.
As far as I can tell, there are ways to get a line-by-line heat map in Jupyter, like [this](https://stackoverflow.com/a/52461806), but you don't get the full call stack like in a flame graph.

![Flame Graph Example](/media/flame-graph.png)

There's a fantastic package called [py-spy](https://github.com/benfred/py-spy) that creates flame graphs, but that works best with .py files.
Now we just need a way to get our Jupyter notebook represented as a .py file.
There are a couple of options for this.
One would be to export the file as a .py file from the File menu of your Jupyter server.
Another option is to use [Jupytext](https://github.com/mwouts/jupytext) to create a synced .py file.
The benefit of this is that if you make changes in the .py file, they'll be reflected in the notebook, making it easier to switch between them.

To do this, first install Jupytext with

```bash
$ pip install jupytext
```

then, if you already have a notebook called `my_notebook.ipynb`, create a synced pair of files with

```bash
$ jupytext --set-formats ipynb,py:percent my_notebook.ipynb
```

which will create and sync `my_notebook.py`.
Go [here](https://jupytext.readthedocs.io/en/latest/using-cli.html) a quick reference of some CLI commands available to Jupytext.

From there, you can run py-spy on the `my_notebook.py`. 
First, install py-spy if you haven't already:

```bash
$ pip install py-spy
```

Then execute the script with py-spy and output a flame graph to `my_flame_graph.svg`:

```bash
$ py-spy record -o my_flame_graph.svg -- python my_notebook.py
```

Drag that SVG file onto a web browser and you can explore your flame graph interactively.

### Some Caveats

If you have Jupyter magic commands (anything starting with `%`) or `display()` invocations, you'll run into errors executing as a flat .py script.
You'll need to comment out the Jupyter magics, and add a `from IPython.display import display` if you have `display()` anywhere.

Please [hit me up on Twitter](https://www.twitter.com/ben_j_lindsay) or comment below if you found this useful, ran into issues with any of the steps in this post, or have suggestions about alternative ways to profile code in Jupyter notebooks.
