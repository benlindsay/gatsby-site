---
title: Parameter Sweep Bash Script
date: "2015-12-19 21:25"
template: "post"
draft: false
slug: "/posts/parameter-sweep-bash-script/"
category: "Projects"
tags:
  - "bash"
  - "productivity"
description: "In my polymer field theory research, often my studies involve
running a bunch of simulations where I pick one or more input parameters and
change them over a range of values, then compare the results of each separate
simulation to see how that/those variable(s) affect the system I’m
simulating. It can be very tedious to manually create input files for each
job, so I wrote a bash script to help me out."
---

In my polymer simulation research, often my studies involve running a bunch
of simulations where I pick one or more input parameters and change them over
a range of values, then compare the results of each separate simulation to
see how that/those variable(s) affect the system I'm simulating. This kind of
study is called a "parameter sweep", and can also be referred to as
"embarrassingly parallel", because the processor(s) for each for each
individual job don't need to communicate with the processor(s) from any other
job. It can be very tedious to manually create input files for each job, so I
wrote a bash script to help me out.

For example, if I want to simulate 3 different polymer nanocomposite systems,
each with a different nanorod length, I could manually create 3 directories
like so:

```shell
mkdir length1
mkdir length2
mkdir length3
```

Then I could copy an input file, `bcp.input`, and a submit file, `sub.sh` into
each of those folders like so:

```shell
for d in length*/ ; do
    cp bcp.input "$d"
    cp sub.sh "$d" 
done
```

Then I could proceed to manually edit all 6 files (or just 3 if the
submission script doesn't have to change). If it's just 3 files, it's not too
bad, but if I want to run 10 or 20 simulations with slight changes in the
input file for each one, manual editing gets real tedious real fast. I got
fed up with it and wrote a script to do all the editing for me. The script is
called `param-sweep.sh`. Feel free to look at it on
[Bitbucket](https://bitbucket.org/benlindsay/param-sweep)

Before running the script, I make a template for the input file and
submission script with parameter names that the script will replace with
parameter values. My input file template could look something like this:

```shell
1000        # Number of iterations
60          # Polymer length
1           # Nanorod radius
NRLENGTH    # Nanorod length
```

and my submission script template could look something like this:

```shell
#!/bin/sh
#PBS -N TRIALNAME
#PBS -l nodes=1:ppn=12
#PBS -l walltime=01:00:00,mem=2gb

cd $PBS_O_WORKDIR

# Run code that looks for bcp.input in the current directory
mpirun $HOME/code/awesome_code.exe
```

In this example, I want to replace `NRLENGTH` with the actual nanorod length
for each `bcp.input` file in `./length1`, `./length2`, and `./length3`, and I
want to replace `TRIALNAME` with a name corresponding to each simulation in
each `sub.sh` file. The script does this by looking through a `trials.txt`
file I make that would look like this in this case:

```
name        i:NRLENGTH  s:TRIALNAME
length1     4           length1-trial
length2     5           length2-trial
length3     6           length3-trial
```

The `i:` and `s:` before `NRLENGTH` and `TRIALNAME`, respectively, tell the
script to look in the input file or submission script for each variable.
Finally, let's look at how to use the script:

```shell
$ ls
bcp.input   trials.txt  sub.sh
$ ~/scripts/param-sweep.sh -t trials.txt -i bcp.input -s sub.sh
Trials file:        trials.txt
Input file:         bcp.input
Submission script:  sub.sh
3 trials
2 vars
Submitting trial length1:
1443364.rrlogin.internal
Submitting trial length2:
1443365.rrlogin.internal
Submitting trial length3:
1443366.rrlogin.internal
$ tree
.
├── bcp.input
├── length1
│   ├── bcp.input
│   └── sub.sh
├── length2
│   ├── bcp.input
│   └── sub.sh
├── length3
│   ├── bcp.input
│   └── sub.sh
├── sub.sh
└── trials.txt
```

So the script made directories for all three simulations, replaced `NRLENGTH`
with `4`, `5`, and `6` in the `bcp.input` files, replaced `TRIALNAME` with
`length1-trial`, `length2-trial`, and `length3-trial` in the `sub.sh` files,
and submitted the `sub.sh` files from within their respective simulation
directories. In this case, since my script expects files with the names I
used, I could have just typed `~/scripts/param-sweep.sh`. If I wanted to be
able to check the files before submitting, I could have typed
`~/scripts/param-sweep.sh -n` which would create the directories and files
without submitting the jobs.

A few caveats: the script isn't currently set up to handle more than one
layer of simulation directories. Also, the script as it's set up right now
copies whatever input file and submission script its fed to files named
`bcp.input` and `sub.sh`. Finally, you'll need to make sure that the variable
name you want the script to find and replace with variable values doesn't
show up anywhere else in the file. The script will find and replace all
instances of the variable name (case sensitive).

This script has saved me a lot of time. Hopefully it can help someone else
out there too.
