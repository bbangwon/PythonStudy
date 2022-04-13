#문제 2의 숫자 n개 중에서 최댓값 찾기를 재귀함수로 만들어보세요

def findmax(mv, v, idx):
    if idx < 0:
        return mv    
    if v[idx] > mv:
        mv = v[idx]
    return findmax(mv, v, idx-1)      

v = [17, 92, 18, 33, 58, 7, 33, 42]
print(findmax(0, v, len(v)-1))

