def make_pair(a):
    n = len(a)
    ret = []
    for i in range(0, n-1):
        for j in range(i + 1, n):
            ret.append(a[i] + " - " + a[j])
    return ret

name = ["Tom", "Jerry", "Mike"]
print(make_pair(name))