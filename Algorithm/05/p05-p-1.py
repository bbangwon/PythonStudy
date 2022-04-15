def fibo(a):
    # 7 = fibo(6) + fibo(5)
    if a <= 1:
        return 0
    elif a == 2:
        return 1
    return fibo(a-1) + fibo(a-2)

print(fibo(7))