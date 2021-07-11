"""
"""
import random
import sys
from random import randint
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd
import time
rand = random_state(random.randrange(sys.maxsize))


class PrivateKey(object):
    def __init__(self, p, q, n):
        self.L = lambda x: (x - 1) // n
        if p == q:
            self.l = p * (p-1)
        else:
            self.l = (p-1) * (q-1)
        try:
            self.m = invert(self.l, n)
        except ZeroDivisionError as e:
            print(e)
            exit()


class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = mpz(rint_round(log2(self.n)))


def generate_prime(bits):
    """Will generate an integer of b bits that is prime using the gmpy2 library  """
    while True:
        possible = mpz(2)**(bits-1) + mpz_urandomb(rand, bits-1)
        if is_prime(possible):
            return possible


def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    # print(p)
    q = generate_prime(bits // 2)
    # print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)


def enc(pub, plain):  # (public key, plaintext) #to do
    return (powmod(pub.g, plain, pub.n_sq) *
            powmod(randint(1, pub.n-1), pub.n, pub.n_sq)) % pub.n_sq


def dec(priv, pub, cipher):  # (private key, public key, cipher) #to do
    return (priv.L(powmod(cipher, priv.l, pub.n_sq))*priv.m) % pub.n


def enc_add(pub, m1, m2):  # to do
    return (enc(pub, m1) * enc(pub, m2)) % pub.n_sq
    """Add one encrypted integer to another"""


def enc_add_const(pub, m, c):  # to do
    return (enc(pub, m)*powmod(pub.g, c, pub.n_sq)) % pub.n_sq
    """Add constant n to an encrypted integer"""


def enc_mul_const(pub, m, c):  # to do
    return powmod(enc(pub, m), c, pub.n_sq)  # (enc(pub, m)**c) % pub.n_sq
    """Multiplies an encrypted integer by a constant"""


if __name__ == '__main__':
    priv, pub = generate_keypair(1024)

    print("****************test dec(),enc()****************")
    print("dec(priv, pub, enc(pub, 18000290)) =",
          dec(priv, pub, enc(pub, 18000290)))

    print("****************test enc_add()****************")
    print("dec(priv, pub, enc_add(pub, 2021, 2000)) =",
          dec(priv, pub, enc_add(pub, 2021, 2000)))

    print("****************test enc_add_const()****************")
    print("dec(priv, pub, enc_add_const(pub, 2021, 4)) =",
          dec(priv, pub, enc_add_const(pub, 2021, 4)))

    print("****************test enc_mul_const()****************")
    print("dec(priv, pub, enc_mul_const(pub, 2021, 4)) =",
          dec(priv, pub, enc_mul_const(pub, 2021, 4)))

    print("****************test 10bits enc_add()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(10)
        b[i] = random.getrandbits(10)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("10bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")
    print("****************test 100bits enc_add()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(100)
        b[i] = random.getrandbits(100)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("100bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")
    print("****************test 1000bits enc_add()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(1000)
        b[i] = random.getrandbits(1000)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("1000bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")

    print("****************test 10bits enc_add_const()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(10)
        b[i] = random.getrandbits(10)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add_const(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("10bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")
    print("****************test 100bits enc_add_const()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(100)
        b[i] = random.getrandbits(100)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add_const(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("100bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")
    print("****************test 1000bits enc_add_const()****************")
    a = [0 for i in range(1000)]
    b = [0 for i in range(1000)]
    c = [0 for i in range(1000)]
    d = [0 for i in range(1000)]
    for i in range(1000):
        a[i] = random.getrandbits(1000)
        b[i] = random.getrandbits(1000)
        c[i] = a[i]+b[i]
    start_time = time.time()
    for i in range(1000):
        d[i] = dec(priv, pub, enc_add_const(pub, a[i], b[i]))
    end_time = time.time()
    for i in range(1000):
        if d[i] != c[i]:
            print("Error!!!")
    print("1000bits for 1000 times!!!\nRunning time %0.2f" %
          (end_time - start_time) + " seconds")

