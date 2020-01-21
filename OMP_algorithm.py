import numpy as np


def get_atom(column, matrix):
    return matrix[:, column]


def normalize(matrix):
    return matrix / np.sqrt(np.sum(matrix ** 2))


def get_basis(A, index):
    basis = np.zeros((A.shape[0], index.shape[0]))
    basis_ptr = 0
    for column in range(0, index.shape[0]):
        basis[:, basis_ptr] = normalize(get_atom(index[column], A))
        basis_ptr = basis_ptr + 1
    return basis


def get_weight(basis, y):
    weight = np.matmul(basis.T, y)
    return weight


def LSP(A_new, y):
    precalc = np.matmul(A_new.T, A_new)
    if A_new.shape[0] == 1 and y.shape[1] == 1:
        A_new_p = np.matmul(np.linalg.inv(np.array([[precalc]])), A_new.T)
    else:
        A_new_p = np.matmul(np.linalg.inv(precalc), A_new.T)
    update = np.matmul(A_new_p, y)
    return update


def get_maxpos(weight):
    pos = 0
    max = 0
    for i in range(0, weight.shape[0]):
        if abs(weight[i, 0]) > max:
            max = abs(weight[i, 0])
            pos = i
    return pos


def get_offset(list, target):
    count = 0
    for num in list:
        if num < target:
            count = count + 1
    return count


def OMP_solve(A, y, threshold):
    A_length = A.shape[1]
    index = np.arange(0, A_length, 1)
    selected_list = []
    x_rec = np.zeros((A_length, 1))
    residue = y
    first_cycle = True
    A_new = np.zeros((A.shape[0], 1))
    while True:
        basis = get_basis(A, index)
        weight = get_weight(basis, residue)
        max_pos = get_maxpos(weight)
        offset = get_offset(selected_list, index[max_pos])
        target = max_pos + offset
        selected_list.append(target)
        index = np.delete(index, [max_pos])
        if first_cycle:
            A_new = A[:, [target]]
            first_cycle = False
        else:
            A_new = np.hstack((A_new, A[:, [target]]))
        lp = LSP(A_new, y)
        ptr = 0
        for num in selected_list:
            x_rec[num, 0] = lp[ptr]
            ptr = ptr + 1
        residue = y - np.matmul(A_new, lp)
        if np.sqrt(np.sum(residue ** 2)) < threshold:
            break
        else:
            print(np.sqrt(np.sum(residue ** 2)))
    return x_rec





if __name__ == "__main__":
    # Test
    x_test = np.random.rand(4, 1)
    A_test = np.random.rand(300000, 4)
    y_test = np.matmul(A_test, x_test)
    x_rec = OMP_solve(A_test, y_test, 1e-4)
    print("------------")
    print(x_test)
    print("------------")
    print(x_rec)
    print("------------")
    print(x_rec - x_test)
    print("------------")
    print(np.sqrt(np.sum(np.square(x_test - x_rec))))
