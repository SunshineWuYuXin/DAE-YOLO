import time
import random
import cv2 as cv
import os
import math
import numpy as np
from skimage import exposure
import shutil


# adjust brightness
def changeLight(img, inputtxt, outputiamge, outputtxt):
    # random.seed(int(time.time()))
    flag = random.uniform(0.5, 1.5)  # Flag>1 means dimming, less than 1 means dimming
    label = round(flag, 2)
    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    outputiamge = os.path.join(outputiamge + "/" + filename + "_" + str(label) + ".png")
    outputtxt = os.path.join(outputtxt + "/" + filename + "_" + str(label) + extension)

    ima_gamma = exposure.adjust_gamma(img, 0.5)

    shutil.copyfile(inputtxt, outputtxt)
    cv.imwrite(outputiamge, ima_gamma)


# Add Gaussian noise
def gasuss_noise(image, inputtxt, outputiamge, outputtxt, mean=0, var=0.01):
    '''
    mean : 均值
    var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise

    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    outputiamge = os.path.join(outputiamge + "/" + filename + "_gasunoise_" + str(mean) + "_" + str(var) + ".png")
    outputtxt = os.path.join(outputtxt + "/" + filename + "_gasunoise_" + str(mean) + "_" + str(var) + extension)

    shutil.copyfile(inputtxt, outputtxt)
    cv.imwrite(outputiamge, out)


# Adjust contrast
def ContrastAlgorithm(rgb_img, inputtxt, outputiamge, outputtxt):
    img_shape = rgb_img.shape
    temp_imag = np.zeros(img_shape, dtype=float)
    for num in range(0, 3):
        in_image = rgb_img[:, :, num]
        Imax = np.max(in_image)
        Imin = np.min(in_image)
        Omin, Omax = 0, 255
        a = float(Omax - Omin) / (Imax - Imin)
        b = Omin - a * Imin
        out_image = a * in_image + b
        out_image = out_image.astype(np.uint8)
        temp_imag[:, :, num] = out_image
    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    outputiamge = os.path.join(outputiamge + "/" + filename + "_contrastAlgorithm" + ".png")
    outputtxt = os.path.join(outputtxt + "/" + filename + "_contrastAlgorithm" + extension)
    shutil.copyfile(inputtxt, outputtxt)
    cv.imwrite(outputiamge, temp_imag)


# rotate
def rotate_img_bbox(img, inputtxt, temp_outputiamge, temp_outputtxt, angle, scale=1):
    nAgree = angle
    size = img.shape
    w = size[1]
    h = size[0]
    for numAngle in range(0, len(nAgree)):
        dRot = nAgree[numAngle] * np.pi / 180
        dSinRot = math.sin(dRot)
        dCosRot = math.cos(dRot)

        nw = (abs(np.sin(dRot) * h) + abs(np.cos(dRot) * w)) * scale
        nh = (abs(np.cos(dRot) * h) + abs(np.sin(dRot) * w)) * scale

        (filepath, tempfilename) = os.path.split(inputtxt)
        (filename, extension) = os.path.splitext(tempfilename)
        outputiamge = os.path.join(temp_outputiamge + "/" + filename + "_rotate_" + str(nAgree[numAngle]) + ".png")
        outputtxt = os.path.join(temp_outputtxt + "/" + filename + "_rotate_" + str(nAgree[numAngle]) + extension)

        rot_mat = cv.getRotationMatrix2D((nw * 0.5, nh * 0.5), nAgree[numAngle], scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # affine transformation
        rotat_img = cv.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv.INTER_LANCZOS4)
        cv.imwrite(outputiamge, rotat_img)

        save_txt = open(outputtxt, 'w')
        f = open(inputtxt)
        for line in f.readlines():
            line = line.split(" ")
            x1 = float(line[0])
            y1 = float(line[1])
            x2 = float(line[2])
            y2 = float(line[3])
            x3 = float(line[4])
            y3 = float(line[5])
            x4 = float(line[6])
            y4 = float(line[7])
            category = str(line[8])
            dif = str(line[9])

            point1 = np.dot(rot_mat, np.array([x1, y1, 1]))
            point2 = np.dot(rot_mat, np.array([x2, y2, 1]))
            point3 = np.dot(rot_mat, np.array([x3, y3, 1]))
            point4 = np.dot(rot_mat, np.array([x4, y4, 1]))
            x1 = round(point1[0], 3)
            y1 = round(point1[1], 3)
            x2 = round(point2[0], 3)
            y2 = round(point2[1], 3)
            x3 = round(point3[0], 3)
            y3 = round(point3[1], 3)
            x4 = round(point4[0], 3)
            y4 = round(point4[1], 3)
            # string = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(x3) + " " + str(y3) + " " + str(x4) + " " + str(y4) + " " + category + " " + dif
            string = str(int(x1)) + " " + str(int(y1)) + " " + str(int(x2)) + " " + str(int(y2)) + " " + str(
                int(x3)) + " " + str(int(y3)) + " " + str(int(x4)) + " " + str(int(y4)) + " " + category + " " + dif
            save_txt.write(string)

#Flip Image
def filp_pic_bboxes(img, inputtxt, outputiamge, outputtxt):
    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    output_vert_flip_img = os.path.join(outputiamge + "/" + filename + "_vert_flip" + ".png")
    output_vert_flip_txt = os.path.join(outputtxt + "/" + filename + "_vert_flip" + extension)
    output_horiz_flip_img = os.path.join(outputiamge + "/" + filename + "_horiz_flip" + ".png")
    output_horiz_flip_txt = os.path.join(outputtxt + "/" + filename + "_horiz_flip" + extension)

    h, w, _ = img.shape
    vert_flip_img = cv.flip(img, 1)
    cv.imwrite(output_vert_flip_img, vert_flip_img)
    horiz_flip_img = cv.flip(img, 0)
    cv.imwrite(output_horiz_flip_img, horiz_flip_img)
    save_vert_txt = open(output_vert_flip_txt, 'w')
    save_horiz_txt = open(output_horiz_flip_txt, 'w')
    f = open(inputtxt)
    for line in f.readlines():
        line = line.split(" ")
        x1 = float(line[0])
        y1 = float(line[1])
        x2 = float(line[2])
        y2 = float(line[3])
        x3 = float(line[4])
        y3 = float(line[5])
        x4 = float(line[6])
        y4 = float(line[7])
        category = str(line[8])
        dif = str(line[9])

        vert_string = str(int(round(w - x1, 3))) + " " + str(int(y1)) + " " + str(int(round(w - x2, 3))) + " " + str(
            int(y2)) + " " + str(int(round(w - x3, 3))) + " " + str(int(y3)) + " " + str(
            int(round(w - x4, 3))) + " " + str(int(y4)) + " " + category + " " + dif
        horiz_string = str(int(x1)) + " " + str(int(round(h - y1, 3))) + " " + str(int(x2)) + " " + str(
            int(round(h - y2, 3))) + " " + str(int(x3)) + " " + str(int(round(h - y3, 3))) + " " + str(
            int(x4)) + " " + str(int(round(h - y4, 3))) + " " + category + " " + dif

        save_horiz_txt.write(horiz_string)
        save_vert_txt.write(vert_string)


# Pan Image
def shift_pic_bboxes(img, inputtxt, outputiamge, outputtxt):
    w = img.shape[1]
    h = img.shape[0]
    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    f = open(inputtxt)
    for line in f.readlines():
        line = line.split(" ")
        x1 = float(line[0])
        y1 = float(line[1])
        x2 = float(line[2])
        y2 = float(line[3])
        x3 = float(line[4])
        y3 = float(line[5])
        x4 = float(line[6])
        y4 = float(line[7])
        category = str(line[8])
        dif = str(line[9])

        x_min = min(x_min, x1, x2, x3, x4)
        y_min = min(y_min, y1, y2, y3, y4)
        x_max = max(x_max, x1, x2, x3, x4)
        y_max = max(y_max, y1, y2, y3, y4)

    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
    y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    if x >= 0 and y >= 0:
        outputiamge = os.path.join(
            outputiamge + "/" + filename + "_shift_" + str(round(x, 3)) + "_" + str(round(y, 3)) + ".png")
        outputtxt = os.path.join(
            outputtxt + "/" + filename + "_shift_" + str(round(x, 3)) + "_" + str(round(y, 3)) + extension)
    elif x >= 0 and y < 0:
        outputiamge = os.path.join(
            outputiamge + "/" + filename + "_shift_" + str(round(x, 3)) + "__" + str(round(abs(y), 3)) + ".png")
        outputtxt = os.path.join(
            outputtxt + "/" + filename + "_shift_" + str(round(x, 3)) + "__" + str(round(abs(y), 3)) + extension)
    elif x < 0 and y >= 0:
        outputiamge = os.path.join(
            outputiamge + "/" + filename + "_shift__" + str(round(abs(x), 3)) + "_" + str(round(y, 3)) + ".png")
        outputtxt = os.path.join(
            outputtxt + "/" + filename + "_shift__" + str(round(abs(x), 3)) + "_" + str(round(y, 3)) + extension)
    elif x < 0 and y < 0:
        outputiamge = os.path.join(
            outputiamge + "/" + filename + "_shift__" + str(round(abs(x), 3)) + "__" + str(round(abs(y), 3)) + ".png")
        outputtxt = os.path.join(
            outputtxt + "/" + filename + "_shift__" + str(round(abs(x), 3)) + "__" + str(round(abs(y), 3)) + extension)

    M = np.float32([[1, 0, x], [0, 1, y]])
    shift_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv.imwrite(outputiamge, shift_img)

    save_txt = open(outputtxt, "w")
    f = open(inputtxt)
    for line in f.readlines():
        line = line.split(" ")
        x1 = float(line[0])
        y1 = float(line[1])
        x2 = float(line[2])
        y2 = float(line[3])
        x3 = float(line[4])
        y3 = float(line[5])
        x4 = float(line[6])
        y4 = float(line[7])
        category = str(line[8])
        dif = str(line[9])
        shift_str = str(int(round(x1 + x, 3))) + " " + str(int(round(y1 + y, 3))) + " " + str(int(round(x2 + x, 3))) + " " + str(int(round(y2 + y, 3))) + " " + str(int(round(x3 + x, 3))) + " " + str(int(round(y3 + y, 3))) + " " + str(int(round(x4 + x, 3))) + " " + str(int(round(y4 + y, 3))) + " " + category + " " + dif
        save_txt.write(shift_str)


# Crop Image
def crop_img_bboxes(img, inputtxt, outputiamge, outputtxt):

    w = img.shape[1]
    h = img.shape[0]
    x_min = 0
    x_max = w
    y_min = 0
    y_max = h
    f = open(inputtxt)
    for line in f.readlines():
        line = line.split(" ")
        x1 = float(line[0])
        y1 = float(line[1])
        x2 = float(line[2])
        y2 = float(line[3])
        x3 = float(line[4])
        y3 = float(line[5])
        x4 = float(line[6])
        y4 = float(line[7])
        category = str(line[8])
        dif = str(line[9])

        x_min = min(x_min, x1, x2, x3, x4)
        y_min = min(y_min, y1, y2, y3, y4)
        x_max = max(x_max, x1, x2, x3, x4)
        y_max = max(y_max, y1, y2, y3, y4)

    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    crop_x_min = int(x_min - random.uniform(0, d_to_left))
    crop_y_min = int(y_min - random.uniform(0, d_to_top))
    crop_x_max = int(x_max + random.uniform(0, d_to_right))
    crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)
    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    (filepath, tempfilename) = os.path.split(inputtxt)
    (filename, extension) = os.path.splitext(tempfilename)
    outputiamge = os.path.join(outputiamge + "/" + filename + "_crop_" + str(crop_x_min) + "_" +
                               str(crop_y_min) + "_" + str(crop_x_max) + "_" +
                               str(crop_y_max) + ".png")
    outputtxt = os.path.join(outputtxt + "/" + filename + "_crop_" + str(crop_x_min) + "_" +
                             str(crop_y_min) + "_" + str(crop_x_max) + "_" +
                             str(crop_y_max) + extension)
    cv.imwrite(outputiamge, crop_img)

    save_txt = open(outputtxt, "w")
    f = open(inputtxt)
    for line in f.readlines():
        line = line.split(" ")
        x1 = float(line[0])
        y1 = float(line[1])
        x2 = float(line[2])
        y2 = float(line[3])
        x3 = float(line[4])
        y3 = float(line[5])
        x4 = float(line[6])
        y4 = float(line[7])
        category = str(line[8])
        dif = str(line[9])
        crop_str = str(int(round(x1 - crop_x_min, 3))) + " " + str(int(round(y1 - crop_y_min, 3))) + " " + str(int(round(x2 - crop_x_min, 3))) + " " + str(int(round(y2 - crop_y_min, 3))) + " " + str(int(round(x3 - crop_x_min, 3))) + " " + str(int(round(y3 - crop_y_min, 3))) + " " + str(int(round(x4 - crop_x_min, 3))) + " " + str(int(round(y4 - crop_y_min, 3))) + " " + category + " " + dif
        save_txt.write(crop_str)


if __name__ == '__main__':
    inputiamge = "dataset/ceshi/images"
    inputtxt = "dataset/ceshi/labelTxt"
    outputiamge = "dataset/ceshi_aug/images"
    outputtxt = "dataset/ceshi_aug/labelTxt"
    angle = [30, 60, 90, 120, 150, 180]
    tempfilename = os.listdir(inputiamge)
    for file in tempfilename:
        (filename, extension) = os.path.splitext(file)
        input_image = os.path.join(inputiamge + "/" + file)
        input_txt = os.path.join(inputtxt + "/" + filename + ".txt")

        img = cv.imread(input_image)

        changeLight(img,input_txt,outputiamge,outputtxt)

        gasuss_noise(img, input_txt, outputiamge, outputtxt, mean=0, var=0.001)

        ContrastAlgorithm(img, input_txt, outputiamge, outputtxt)

        rotate_img_bbox(img, input_txt, outputiamge, outputtxt, angle)

        filp_pic_bboxes(img, input_txt, outputiamge, outputtxt)

        shift_pic_bboxes(img, input_txt, outputiamge, outputtxt)

        crop_img_bboxes(img, input_txt, outputiamge, outputtxt)
    print("finished!")
