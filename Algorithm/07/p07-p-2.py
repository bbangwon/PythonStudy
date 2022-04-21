def student(snos, snames, no):
    n1 = len(snos)
    n2 = len(snames)

    for i in range(0, n1):
        if snos[i] == no and i < n2:
            return snames[i]
    
    return "?"

stu_no = [39, 14, 67, 105]
stu_name = ["Justin", "Jhon", "Mike", "Summer"]

print(student(stu_no, stu_name, 39))
print(student(stu_no, stu_name, 14))
print(student(stu_no, stu_name, 99))

