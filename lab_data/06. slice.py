import numpy as np

a = [1, 2, 3, 4, 5]

print(a[1:4])
print(a[:-3])
print(a[2:])
print(a[:])

b = np.array(
    [
        ['00', '01', '02', '03'],
        ['10', '11', '12', '13'],
        ['20', '21', '22', '23'],
        ['30', '31', '32', '33'],
    ]
)
print(b)

print(b[1:3, 1:])
print(b[:2, :])
