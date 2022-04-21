def search(a, v):
    n = len(a)
    r = []
    for i in range(0, n):
        if a[i] == v:
            r.append(i)
    
    return r

v = [17, 92, 18, 33, 58, 7, 33, 42]
print(search(v, 18))
print(search(v, 33))
print(search(v, 100))
