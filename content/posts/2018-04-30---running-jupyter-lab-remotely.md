---
title: "Running Jupyter Lab Remotely"
date: "2018-04-30"
template: "post"
slug: "/posts/running-jupyter-lab-remotely"
category: "Data Science"
tags:
  - "bash"
  - "jupyter"
  - "tmux"
  - "ssh"
  - "productivity"
description: "I'm a huge fan of Jupyter Notebooks, and
I was very excited when I found out about Jupyter Lab, which provides a much
more comprehensive user experience around Jupyter Notebooks. Here I share how
to run Jupyter Lab efficiently on a remote machine. I have a research cluster
where I do most of my analyses for my PhD work, and running Jupyter Lab
directly on the cluster means I don't have to copy files between the cluster
and my desktop."
---

I'm a huge fan of [Jupyter Notebooks](http://jupyter.org/), and I was very
excited when I found out about Jupyter Lab, which provides a much more
comprehensive user experience around Jupyter Notebooks.
[Other](https://towardsdatascience.com/jupyter-notebooks-are-breathtakingly-featureless-use-jupyter-lab-be858a67b59d)
[posts](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906)
have covered in more detail why we should switch to using Jupyter Lab
instead, so I won't talk about that here.

Instead, I just want to share how to run Jupyter Lab efficiently on a remote
machine. I have a research cluster where I do most of my analyses for my PhD
work, and running Jupyter Lab directly on the cluster means I don't have to
copy files betw een the cluster and my desktop.

## The basic commands

To run Jupyter Lab on a remote machine, you need to open 2 terminal windows.
In the first window:

```shell
$ ssh username@hostname
$ jupyter lab --no-browser --port=8888
...
[I 10:17:14.160 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 10:17:14.160 LabApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Then in the second window:

```shell
$ ssh -Y -N -L localhost:8888:localhost:8888 username@hostname
```

Then in your web browser of choice, copy

```
http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

in the url bar. You could just copy `localhost:8888`, but then it might ask
for your token, which you'd have to copy and paste from your remote machine.

All that is kind of a lot just to open up Jupyter Lab. So I found ways to
significantly simplify the process from both the remote and local side.

## Simplfying the remote side

To make things easier on the remote machine side of things, `tmux` (or
`screen`) and aliases really come in handy. I like to have as Jupyter Lab
session running constantly in my remote machine whether I'm logged in or not.
Then I can just ssh tunnel in to the existing session whenever I want! To do
this, I simply do the following:

```shell
$ ssh username@hostname
$ tmux
[ opens persistent shell session ]
$ jlremote
...
[I 10:17:14.160 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 10:17:14.160 LabApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=b6ff64ea67275581a2ec91790e1ed591c945f30ee0f7a214
```

I have `jlremote` defined as an alias in my remote `~/.bashrc` file like so:

```shell
alias jlremote='jupyter lab --no-browser --port=8888'
```

So once I have that Jupyater Lab session running, I can detach from the tmux
session with `CTRL-b, d` (or `CTRL-a, CTRL-d` if you used the `screen`
command), and let that process run indefinitely (days, weeks, months...).

Now let's deal with the local stuff.

## Simplfying the local side

On the local side, I wanted to be able to just run a single command like
`jllocal` to open Jupyter Lab, so I wrote a bash function that goes in my
local `~/.bashrc` file. If you use this make sure to edit all the all-caps
stuff, like `USERNAME`, `HOSTNAME`, and `REMOTE/PATH/TO/jupyter`. The Jupyter
path should be something like `~/anaconda3/bin/jupyter`.

```shell
function jllocal {
  cmd="ssh -Y -fN -L localhost:8888:localhost:8888 USERNAME@HOSTNAME"
  running_cmds=$(ps aux | grep -v grep | grep "$cmd")
  if [[ "$1" == 'kill' ]]; then
    if [ ! -z $running_cmds ]; then
      for pid in $(echo $running_cmds | awk '{print $2}'); do
        echo "killing pid $pid"
        kill -9 $pid
      done
    else
      echo "No jllocal commands to kill."
    fi
  else
    if [ ! -z $n_running_cmds ]; then
      echo "jllocal command is still running. Kill with 'jllocal kill' next time."
    else
      echo "Running command '$cmd'"
      eval "$cmd"
    fi
    url=$(ssh USERNAME@HOSTNAME \
            '/REMOTE/PATH/TO/jupyter notebook list' \
            | grep http | awk '{print $1}')
    echo "URL that will open in your browser:"
    echo "$url"
    open "$url"
  fi
}
```

This function does a few things when you type `jllocal`:

1. Runs ssh tunneling command if it's not already running
2. Grabs the Jupyter token from the remote machine
3. Opens a tab in your browser with the right url and token for you

When you're done with Jupyter Lab, you just type `jllocal kill` and it will
shut down the ssh connection.

## Putting it all together

So with a simple alias in place in your remote `~/.bashrc`, a persistent
remote tmux/screen session running Jupyter Lab, and a function defined in
your local `~/.bashrc`, all you need to do to open Jupyter Lab in your
browser is a simple `jllocal` on your local machine, and then `jllocal kill`
when you're done. It takes some initial set up work, but the simplicity in
the end is worth it.
