# 리스트에서 특정 숫자 위치 찾기(이분 탐색)
# 입력: 리스트 a, 찾는 값 x
# 출력: 찾으면 그 값의 위치, 찾지 못하면 -1

def binary_search(a, x, start=0, end=-1):
    # 탐색대상이 비어있을 경우 종료한다.
    if end == -1:
        end = len(a)

    if start == end:
        return -1

    center = (start + end) // 2
    if a[center] == x:
        return center
    elif a[center] > x:
        # 왼쪽
        return binary_search(a, x, start, center)
    else:
        # 오른쪽
        return binary_search(a, x, center+1, end)


d = [1, 4, 9, 16, 25, 36, 49, 64, 81]
print(binary_search(d, 36))
print(binary_search(d, 50))
