import numpy as np

rng = np.random.default_rng()

from dass.analysis import ES

"""
Bayes' theorem states that
p(x|y) is proportional to p(y|x)p(x)

Assume p(x) is N(0, 2I) and p(y|x) is N(y, 2I).
Multiplying these together (see 8.1.8 of the matrix cookbook) we get
that p(x|y) is N(y/2, I).

Here we use this property, and assume that the forward model is the identity
to test analysis steps.
"""

N = 1000
nparam = 3
m = nparam
var = 2

# A is p(x)
A = rng.multivariate_normal(
    mean=np.zeros(nparam), cov=var * np.identity(nparam), size=(N)
).T
# Assuming forward model is the identity
Y = A

Cdd = var * np.identity(m)
E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
observation = 10.0
D = np.ones((m, N)) * observation + E


def test_ES():
    X = ES(Y, D, Cdd)

    A_ES = A @ X

    # Potentially flaky, but still useful.
    assert np.isclose(A_ES[0, :].mean(), observation / 2, rtol=0.1)
    assert np.isclose(A_ES[1, :].mean(), observation / 2, rtol=0.1)
    assert (np.abs(np.cov(A_ES) - np.identity(nparam)) < 0.15).all()


def test_ES_with_localisation():
    A_ES_local = A.copy()
    for i in range(nparam):
        X_local = ES(Y, D, Cdd)
        A_ES_local[i, :] = A_ES_local[i, :] @ X_local

    X = ES(Y, D, Cdd)
    A_ES = A @ X

    assert np.isclose(A_ES_local, A_ES).all()
