from PIL import Image
import numpy as np


DIR_FORM = "./AR/{}-{}-{}.pgm"
# E.G. m-001-01.pgm 165x120 255
# m: male   w: female
# 001 - 050: distinct people
# 01 - 26: distinct poses
# 01 - 07 & 14 - 20: training set
# 08 - 13 & 21 - 26: testing set


def calc_train(block=False):
    if block:
        # Every trait matrix has  rows and 14 columns
        person_trait1 = np.zeros((140, 1400))
        person_trait2 = np.zeros((140, 1400))
        person_trait3 = np.zeros((140, 1400))
        person_trait4 = np.zeros((140, 1400))
        person_trait5 = np.zeros((140, 1400))
        person_trait6 = np.zeros((140, 1400))
        person_trait7 = np.zeros((140, 1400))
        person_trait8 = np.zeros((140, 1400))
        person_trait9 = np.zeros((140, 1400))
        column_ptr = 0
        for person in range(1, 51):
            for pose in range(1, 27):
                if pose <= 7 or (14 <= pose <= 20):
                    file_dir = DIR_FORM.format("m", str(person).zfill(3), str(pose).zfill(2))
                    person_trait_list = process_img(file_dir, True)
                    person_trait1[:, column_ptr] = person_trait_list[0]
                    person_trait2[:, column_ptr] = person_trait_list[1]
                    person_trait3[:, column_ptr] = person_trait_list[2]
                    person_trait4[:, column_ptr] = person_trait_list[3]
                    person_trait5[:, column_ptr] = person_trait_list[4]
                    person_trait6[:, column_ptr] = person_trait_list[5]
                    person_trait7[:, column_ptr] = person_trait_list[6]
                    person_trait8[:, column_ptr] = person_trait_list[7]
                    person_trait9[:, column_ptr] = person_trait_list[8]
                    column_ptr = column_ptr + 1
        for person in range(1, 51):
            for pose in range(1, 27):
                if pose <= 7 or (14 <= pose <= 20):
                    file_dir = DIR_FORM.format("w", str(person).zfill(3), str(pose).zfill(2))
                    person_trait_list = process_img(file_dir, True)
                    person_trait1[:, column_ptr] = person_trait_list[0]
                    person_trait2[:, column_ptr] = person_trait_list[1]
                    person_trait3[:, column_ptr] = person_trait_list[2]
                    person_trait4[:, column_ptr] = person_trait_list[3]
                    person_trait5[:, column_ptr] = person_trait_list[4]
                    person_trait6[:, column_ptr] = person_trait_list[5]
                    person_trait7[:, column_ptr] = person_trait_list[6]
                    person_trait8[:, column_ptr] = person_trait_list[7]
                    person_trait9[:, column_ptr] = person_trait_list[8]
                    column_ptr = column_ptr + 1
        return [person_trait1, person_trait2, person_trait3,
                person_trait4, person_trait5, person_trait6,
                person_trait7, person_trait8, person_trait9]
    else:
        # Every trait matrix has 1260 rows and 14 columns
        person_trait = np.zeros((1260, 1400))
        column_ptr = 0
        for person in range(1, 51):
            for pose in range(1, 27):
                if pose <= 7 or (14 <= pose <= 20):
                    file_dir = DIR_FORM.format("m", str(person).zfill(3), str(pose).zfill(2))
                    person_trait[:, column_ptr] = process_img(file_dir)
                    column_ptr = column_ptr + 1
        for person in range(1, 51):
            for pose in range(1, 27):
                if pose <= 7 or (14 <= pose <= 20):
                    file_dir = DIR_FORM.format("w", str(person).zfill(3), str(pose).zfill(2))
                    person_trait[:, column_ptr] = process_img(file_dir)
                    column_ptr = column_ptr + 1
        return person_trait


def calc_person_test(gender, person):
    # Every trait matrix has 1260 rows and 12 columns
    person_trait = np.zeros((1260, 12))
    column_ptr = 0
    for pose in range(1, 27):
        if 8 <= pose <= 13 or 21 <= pose:
            file_dir = DIR_FORM.format(gender, str(person).zfill(3), str(pose).zfill(2))
            person_trait[:, column_ptr] = process_img(file_dir)
            column_ptr = column_ptr + 1
    return person_trait


def process_img(file_dir, block=False):
    img = read_img(file_dir)
    img_down = down_sampling(img)
    # if we gonna divided the img into different blocks,
    # say 9 blocks, each block is 14x10
    if block:
        img1, img2, img3 = np.hsplit(img_down, 3)
        img11, img12, img13 = np.vsplit(img1, 3)
        img21, img22, img23 = np.vsplit(img2, 3)
        img31, img32, img33 = np.vsplit(img3, 3)
        ir11 = img11.reshape((1, 140))
        ir12 = img12.reshape((1, 140))
        ir13 = img13.reshape((1, 140))
        ir21 = img21.reshape((1, 140))
        ir22 = img22.reshape((1, 140))
        ir23 = img23.reshape((1, 140))
        ir31 = img31.reshape((1, 140))
        ir32 = img32.reshape((1, 140))
        ir33 = img33.reshape((1, 140))

        in11 = ir11 / np.sqrt(np.sum(ir11 ** 2))
        in12 = ir12 / np.sqrt(np.sum(ir12 ** 2))
        in13 = ir13 / np.sqrt(np.sum(ir13 ** 2))
        in21 = ir21 / np.sqrt(np.sum(ir21 ** 2))
        in22 = ir22 / np.sqrt(np.sum(ir22 ** 2))
        in23 = ir23 / np.sqrt(np.sum(ir23 ** 2))
        in31 = ir31 / np.sqrt(np.sum(ir31 ** 2))
        in32 = ir32 / np.sqrt(np.sum(ir32 ** 2))
        in33 = ir33 / np.sqrt(np.sum(ir33 ** 2))
        return [in11, in12, in13, in21, in22, in23, in31, in32, in33]
    else:
        # 42 x 30 = 1260
        img_reshape = img_down.reshape((1, 1260))
        # Normalization
        img_norm = img_reshape / np.sqrt(np.sum(img_reshape ** 2))
        return img_norm


def read_img(file_name):
    im = Image.open(file_name)
    img = np.array(im)
    return img


# original footage is 165x120
# down-sample to 42x30 (nearly 4x4)
def down_sampling(img):
    return (img[:, range(0, img.shape[1], 4)])[range(0, img.shape[0], 4), :]


if __name__ == "__main__":
    calc_train()