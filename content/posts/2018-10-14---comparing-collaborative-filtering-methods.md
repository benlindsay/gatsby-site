---
title: "Comparing Collaborative Filtering Methods"
date: "2019-10-14"
template: "post"
draft: true
slug: "/posts/comparing-collaborative-filtering-methods"
category: "Projects"
tags:
  - "recommender systems"
  - "collaborative filtering"
  - "python"
  - "numpy"
  - "pandas"
  - "seaborn"
  - "jupyter"
  - "matplotlib"
description: "I wanted to dive into the fundamentals of collaborative
filtering and recommender systems, so I implemented a few common methods and
compared them."
---

As part of a project sponsored by the data science team at [Air
Liquide](https://www.airliquide.com/) here in Philadelphia, I'm diving deep
into collaborative filtering algorithms. There are 2 main goals of this
project:

1. Gain understanding of a variety of collaborative filtering algorithms by implementing them myself
2. Compare quality and speed of a variety of algorithms as a function of dataset
   size

The data I'm using comes from the [GroupLens](https://grouplens.org/)
research group, which has been collecting movie ratings from volunteers since
1995 and has curated datasets of a variety of sizes. For simplicity, I'll
focus on the 100K dataset, the smallest one, to enable faster iteration.

I split this project into several parts. Here's a table of contents for you:

- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Baseline Algorithms](#baseline-algorithms)
  - [Simple Average Model](#simple-average-model)
  - [Average By ID Model](#average-by-id-model)
  - [Damped User + Movie Baseline](#damped-user--movie-baseline)
  - [Baseline Comparison](#baseline-comparison)
- [Similarity-Based Algorithms](#similarity-based-algorithms)
- [Alternating Least Squares](#alternating-least-squares)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Algorithm Comparisons](#algorithm-comparisons)
- [Recommender System Prototype](#recommender-system-prototype)

## Exploratory Data Analysis

*Check out the full notebook for this section*
*[here](https://github.com/benlindsay/movielens-analysis/blob/master/01_Exploratory-Analysis-on-100K-data.ipynb).*

Before getting into any algorithm development, I wanted to get a picture of the
data I was working with, so I asked the questions on my mind and tried to
answer them with the data.

What does the ratings distribution look like?

![Ratings Distribution](/media/movielens-ratings-distribution.png)

It's a little skewed to the positive side, with 4 being the most common rating.
I guess that skew makes sense because people are more likely to watch stuff they
would like than garbage they would hate.

Next: how consistent are the ratings over time? If people as a whole get more
positive or negative over time, that could complicate things. If their behavior
doesn't seem to change too much, we can make a simplifying assumption that time
doesn't matter and ignore time dependence.

![Ratings Consistency](/media/movielens-ratings-consistency.png)

Looks pretty consistent, so we're going to make that simplifying assumption.
Purely out of curiosity, how much do the number of users and movies change over
time?

![User and Movie Count](/media/movielens-movie-and-user-count.png)

The amount of growth in the short timespan of this dataset, particularly in the
number of users, does make me think a more complicated approach could be
warranted. Buuuuuut I don't want to do that right now. We'll stick with assuming
we're working with an
[IID](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
dataset for the purposes of this project.

A very crucial aspect to understand about typical recommendation situations is
the sparsity of your dataset. You want to predict how much every user likes
every movie, but we have data about very few user-movie combinations. We'll
explore this in two ways.

First we'll visualize the sparsity pattern of the user-movie matrix. This
could be done with Matplotlib's
[spy](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.spy.html)
function, but I didn't know about it at the time I did this analysis, so I
did this manually. The plot below shows a single, tiny black square for every
user/movie combination we have. If everyone rated every movie, you'd see a
solid black rectangle. Instead what we see is a lot of white--lots of
user/movie combinations for which we don't have a rating (yet). You
especially see a lot of white in the top right corner. This is probably
because early raters had access to fewer movies to rate, and new users
progressively had more movies to rate as they were added to the system.

![MovieLens Sparsity Map](/media/movielens-sparsity-map.png)

The matrix density is $n_{ratings}/(n_{users}×n_{movies})=0.063$, meaning that
about 94% of the data we would like to know is missing.

In the plot above you also notice that there are a few darker rows and
columns, but most rows and columns are pretty bare. Let's visualize the
distributions of number of ratings by user and by movie. The way I chose to
visualize this is with an [Empirical Cumulative Distribution Function
(ECDF)](https://en.wikipedia.org/wiki/Empirical_distribution_function) plot.
An ECDF plot has an advantage compared to a histogram that all data points
can be plotted in a meaningful way, and no bin size has to be chosen to
average arbitrary chunks of it. This is especially helpful with the
long-tailed distributions here.

![ECDF Plot](/media/movielens-ecdf.png)

In the plot above, you can learn, for example, that 40% of all users rated 50
or less movies, and 90% of movies have 169 or less ratings. In general, we
seen that a large fraction of movies and users have few ratings associated
with them, but a few movies and users have many more ratings.

The main thing to take from this though is that the matrix of possible
ratings is quite sparse, and that we need to use models that deal with this
lack of data.

## Baseline Algorithms

*Check out the full notebook for this section*
*[here](https://github.com/benlindsay/movielens-analysis/blob/master/02_Baselines.ipynb).*

Baseline models are important for 2 key reaons:

1. Baseline models give us a starting point to which to compare all future
   models, and
2. Smart baselines/averages may be needed to fill in missing data for more
   complicated models

In this section, we'll explore a few typical baseline models for recommender
systems and see which ones do the best for our dataset. For all of these baseline
models, and for that matter all the "real" models in the following sections, I
coded them with the following structure, roughly similar to Scikit-learn's API:

```python
class SomeBaselineModel():

    def __init__(self):
        # Run initialization steps

    def fit(self, X):
        # Compute model parameters from ratings dataframe X with user, movie,
        # and rating columns
        ...
        return self

    def predict(self, X):
        # Predict ratings for dataframe X with user and movie columns
        ...
        return predictions
```

I won't actually put the code for all the models in here, but it's all there in
the Jupyter notebook.

### Simple Average Model

The first model I implemented is about the simplest one possible, which I called
`SimpleAverageModel`. We'll average all the training set ratings and use that
average for the prediction for all test set examples. It probably won't do very
well, but hey, it's a baseline!

### Average By ID Model

We can probably do a little better by using the user or item (movie) average. To
do this, I set up a baseline model class, which I called `AverageByIdModel`,
that allows you to pass either a list of `userId`s or `movieId`s as `X`. The
prediction for a given ID will be the average of ratings from that ID, or the
overall average if that ID wasn't seen in the training set. This will
probably get us a little farther than `SimpleAverageModel` but it still won't
win any million-dollar prizes.

### Damped User + Movie Baseline

Lastly, we can likely do even better by taking into account average user **and**
movie data for a given user-movie combo. It has an additional feature of a
damping factor that can regularize the baseline prediction to prevent us from
straying too far from that average of 4. The damping factor has been shown
empirically to improve the baseline's perfomance. I called my implementation
`DampedUserMovieBaselineModel`. 

This model follows equation 2.1 from a [collaborative filtering
paper](http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf) from
[GroupLens](https://grouplens.org/), the same group that published the
MovieLens data. This equation defines rhe baseline rating for user $u$ and
item $i$ as

$$b_{u,i} = \mu + b_u + b_i$$

where

$$b_u = \frac{1}{|I_u| + \beta_u}\sum_{i \in I_u} (r_{u,i} - \mu)$$

and

$$b_i = \frac{1}{|U_i| + \beta_i}\sum_{u \in U_i} (r_{u,i} - b_u - \mu).$$

(See equations 2.4 and 2.5). Here, $\beta_u$ and $\beta_i$ are damping
factors, for which the paper reported 25 is a good number for this dataset.
For now we'll just leave these values equal ($\beta=\beta_u=\beta_i$). Here's
a summary of the meanings of all the variables here:

| Term            | Meaning                                               |
|:--------------- |:----------------------------------------------------- |
| $b_{u,i}$       | Baseline rating for user $u$ on item (movie) $i$      |
| $\mu$           | The mean of all ratings                               |
| $b_u$           | The deviation from $\mu$ associated with user $u$     |
| $b_i$           | The deviation from $\mu+b_u$ associated with user $i$ |
| $I_u$           | The set of all items rated by user $u$                |
| $\mid I_u \mid$ | The number of items rated by user $u$                 |
| $\beta_u$       | Damping factor for the users ($=\beta$)               |
| $r_{u,i}$       | Observed rating for user $u$ on item $i$              |
| $U_i$           | The set of all users who rated item $i$               |
| $\mid U_i \mid$ | The number of users who rated item $i$                |
| $\beta_i$       | Damping factor for the items ($=\beta$)               |

### Baseline Comparison

With those baseline models defined, let's compare them. In the plot below, I
test 7 baseline models. The first is the `SimpleAverageModel`. The next two use
the `AverageByIdModel` looking at averages by Item ID and User ID, respectively.
The last 4 use the `DampedUserMovieBaseline` with different damping factors
($\beta$). The top plot shows the Mean Absolute Error (MAE) of each fold after
using 5-fold cross-validation. I chose MAE so as not to overly penalize more
extreme ratings (compared to Mean Squared Error) from people angrily or
over-excitedly selecting 1 or 5. The bottom plot shows the distributions of the
corresponding residuals, meaning the difference between actual and predicted
ratings.

![Baseline Comparison](/media/movielens-baseline-comparison.png)

The MAE plots above show that the combined model with a damping factor of 0
or 10 performs the best, followed by the item average, then the user average.
It makes sense that taking into account deviations from the mean due to both
user and item would perform the best: there are more degrees of freedom
($n_{users}+n_{movies}$ to be exact) being taken into account for each baseline
prediction. The same idea explains why the item average performs better than the
user average: there are more items than users in this dataset, so averaging over
items gives you $n_{movies}$ degrees of freedom, which is greater than the
$n_{users}$ degrees of freedom for the user average. The residual plots
underneath the MAE plot illustrate that taking into account more data pulls
the density of the residuals closer to 0.

Before moving on to collaborative filtering models, we'll want to choose
which model to use as a baseline. Both the Combined 0 and Combined 10 models
performed equally well, but we'll choose the Combined 10 model, because a higher
damping factor is effectively stronger regularization, which will prevent
overfitting better than a damping factor of 0.

## Similarity-Based Algorithms

## Alternating Least Squares

## Stochastic Gradient Descent

## Algorithm Comparisons

## Recommender System Prototype
