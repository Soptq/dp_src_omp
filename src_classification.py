import src_import, OMP_algorithm
from matplotlib.pyplot import *
import time


DIR_FORM = "./AR/{}-{}-{}.pgm"


def one_hoted(x, threohold):
    counter = 0
    for pose in range(0, 1400):
        if x[pose, 0] < threohold:
            x[pose, 0] = 0
        else:
            x[pose, 0] = 1
            counter = counter + 1
    return x, counter


def get_residue(A, x, y, n):
    x_n = np.zeros(shape=x.shape)
    for i in range(14*(n - 1), 14*n):
        x_n[i, 0] = x[i, 0]
    return np.sqrt(np.sum(np.square(y - np.matmul(A, x_n))))


def identified(A, x, y):
    residue = 1e10
    iden_p = -1
    ax = np.arange(100)
    ay = np.arange(100)
    for person in range(0, 100):
        ay[person] = get_residue(A, x, y, person)
        if get_residue(A, x, y, person) < residue:
            residue = get_residue(A, x, y, person)
            iden_p = person
    # draw_bar(ax, ay)
    return residue, iden_p


def draw_bar(x, y):
    bar(x, +y, facecolor='#9999ff', edgecolor='white')
    show()


def testify_img_src(A, answer, file_dir, threshold):
    print("TESTIFTING {} ......".format(file_dir))
    y = src_import.process_img(file_dir)
    x = OMP_algorithm.OMP_solve(A, y.T, threshold)
    r, p = identified(A, x, y)
    print(answer == p)
    return answer == p


def testify_img_block(A_list, answer, file_dir, threshold, borda=False):
    print("TESTIFYING {} ......".format(file_dir))
    y_list = src_import.process_img(file_dir, True)
    x1 = OMP_algorithm.OMP_solve(A_list[0], y_list[0].T, threshold)
    x2 = OMP_algorithm.OMP_solve(A_list[1], y_list[1].T, threshold)
    x3 = OMP_algorithm.OMP_solve(A_list[2], y_list[2].T, threshold)
    x4 = OMP_algorithm.OMP_solve(A_list[3], y_list[3].T, threshold)
    x5 = OMP_algorithm.OMP_solve(A_list[4], y_list[4].T, threshold)
    x6 = OMP_algorithm.OMP_solve(A_list[5], y_list[5].T, threshold)
    x7 = OMP_algorithm.OMP_solve(A_list[6], y_list[6].T, threshold)
    x8 = OMP_algorithm.OMP_solve(A_list[7], y_list[7].T, threshold)
    x9 = OMP_algorithm.OMP_solve(A_list[8], y_list[8].T, threshold)
    r1, p1 = identified(A_list[0], x1, y_list[0])
    r2, p2 = identified(A_list[1], x2, y_list[1])
    r3, p3 = identified(A_list[2], x3, y_list[2])
    r4, p4 = identified(A_list[3], x4, y_list[3])
    r5, p5 = identified(A_list[4], x5, y_list[4])
    r6, p6 = identified(A_list[5], x6, y_list[5])
    r7, p7 = identified(A_list[6], x7, y_list[6])
    r8, p8 = identified(A_list[7], x8, y_list[7])
    r9, p9 = identified(A_list[8], x9, y_list[8])
    if borda:
        iden_list = [[r1, p1], [r2, p2], [r3, p3], [r4, p4], [r5, p5],
                     [r6, p6], [r7, p7], [r8, p8], [r9, p9]]
        iden_list.sort(key=lambda x: x[0])
        M = 9
        borda_list = [0 for i in range(100)]
        for iden in iden_list:
            borda_list[iden[1] - 1] = borda_list[iden[1] - 1] + M
            M = M - 1
        return borda_list.index(max(borda_list)) + 1 == answer
    else:
        iden_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        print(np.argmax(np.bincount(iden_list)) == answer)
        return np.argmax(np.bincount(iden_list)) == answer


def get_accuracy_src(A, threshold):
    right_counter = 0
    global_counter = 0
    for person in range(1, 51):
        for pose in range(1, 27):
            # if 8 <= pose <= 13 or 21 <= pose:
            if 8 <= pose <= 13:
                global_counter = global_counter + 1
                file_dir = DIR_FORM.format("m", str(person).zfill(3), str(pose).zfill(2))
                if testify_img_src(A, person, file_dir, threshold):
                    right_counter = right_counter + 1
                file_dir = DIR_FORM.format("w", str(person).zfill(3), str(pose).zfill(2))
                if testify_img_src(A, 50 + person, file_dir, threshold):
                    right_counter = right_counter + 1
                print("CURRENT ACCURACY: {}".format(right_counter/global_counter))
    return right_counter/global_counter


def get_accuracy_block(A_list, threshold, borda=False):
    if borda:
        print("------ USING BORDA VOTER ------")
    right_counter = 0
    global_counter = 0
    for person in range(1, 51):
        for pose in range(1, 27):
            if 8 <= pose <= 13 or 21 <= pose:
            # if 8 <= pose <= 13:
                global_counter = global_counter + 1
                file_dir = DIR_FORM.format("m", str(person).zfill(3), str(pose).zfill(2))
                if borda:
                    if testify_img_block(A_list, person, file_dir, threshold, True):
                        right_counter = right_counter + 1
                else:
                    if testify_img_block(A_list, person, file_dir, threshold):
                        right_counter = right_counter + 1
                print("CURRENT ACCURACY: {}".format(right_counter / global_counter))
                global_counter = global_counter + 1
                file_dir = DIR_FORM.format("w", str(person).zfill(3), str(pose).zfill(2))
                if borda:
                    if testify_img_block(A_list, person, file_dir, threshold, True):
                        right_counter = right_counter + 1
                else:
                    if testify_img_block(A_list, 50 + person, file_dir, threshold):
                        right_counter = right_counter + 1
                print("CURRENT ACCURACY: {}".format(right_counter / global_counter))
    return right_counter/global_counter


if __name__ == "__main__":
    A_list = src_import.calc_train(True)
    # A = src_import.calc_train()
    # print(testify_img_block(A_list, 1, "./AR/m-001-08.pgm", 1e-1))
    # print(get_accuracy_block(A_list)/1200)
    # print(get_accuracy_src(A, 1e-1))
    time_start = time.time()
    ab = get_accuracy_block(A_list, 1e-1)
    time_end = time.time()
    ab_c = time_end - time_start
    # print(get_accuracy_src(A, 1e-1))
    time_start = time.time()
    abb = get_accuracy_block(A_list, 1e-1, True)
    time_end = time.time()
    abb_c = time_end - time_start
    print("---------- RESULT ----------")
    print("")
    print("accuracy:", ab)
    print("time cost:", ab_c)
    print("")
    print("accuracy:", abb)
    print("time cost:", abb_c)
    print("----------- END ------------")