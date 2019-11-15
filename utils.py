import random


def pipeline(*args):
    def p(x):
        for f in args:
            x = f(x)

        return x

    return p


def pick_random(xs):
    return xs[random.randrange(len(xs))]
