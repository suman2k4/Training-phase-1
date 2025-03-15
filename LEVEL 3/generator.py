def fib(num_terms):
    
    a,b=0,1

    for i in range(num_terms):
        yield a
        a, b = b, a + b

fibonacci_seqence=list(fib(5))

print(fibonacci_seqence)