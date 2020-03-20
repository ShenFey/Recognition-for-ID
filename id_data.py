from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import random

NUMBER_SIZE = 18

for i in range(1, 500):
    def txt_setting(image, size, draw_x, draw_y, txt):
        """声明写字函数"""
        # 字体字号
        set_font = ImageFont.truetype('simhei.ttf', size)
        # 创建一个对象，可以在 image 上绘画
        draw = ImageDraw.Draw(image)
        # 调用方法
        draw.text((draw_x, draw_y), txt, font=set_font, fill=(0, 0, 0))
        return image


    def make_number():
        # 随机生成身份证号码
        text_num = ''
        card_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for x in range(NUMBER_SIZE):
            num = random.choice(card_number)
            text_num = text_num + num
        else:
            return text_num


    def make_white_mask():
        # 生成一个空白的模板 mask
        ori_image = cv2.imread('id_card1.png')
        # 大小为原尺寸的0.4倍
        ori_image = cv2.resize(ori_image, (0, 0), fx=0.4, fy=0.4)
        # 返回一个用1填充，形状，类型和原来大小一致的数组
        mask_image = np.ones_like(ori_image)
        # 赋值为255 （白色模板）
        mask_image *= 255

        # 往空白模板上写字(这里只能用PIL写，因为OpenCV写中文会乱码)
        img = Image.fromarray(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
        text = make_number()
        img = txt_setting(img, 18, 145, 234, text)

        mask_image_txt = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 把模板灰度化
        gray = cv2.cvtColor(mask_image_txt, cv2.COLOR_BGR2GRAY)

        # 高斯模糊，制造边缘模糊效果，方便滤除噪声
        gray_gaussianblur = cv2.GaussianBlur(gray, (3, 3), 0)
        # cv2.imwrite('test/gaussianblur.jpg', gray_gaussianblur)

        # 使用阈值对图片进行二值化
        thresh, result = cv2.threshold(gray_gaussianblur, 200, 255, cv2.THRESH_BINARY)
        res_inv = cv2.bitwise_not(result)

        # 写字的模板保留文字部分
        img_bg = cv2.bitwise_and(mask_image_txt, mask_image_txt, mask=res_inv)
        # 原图保留除文字的其他部分
        img_fg = cv2.bitwise_and(ori_image, ori_image, mask=result)
        # 将两张图直接进行相加，即可
        final = cv2.add(img_bg, img_fg)
        path = '../picture/final{}.jpg'.format(i)
        cv2.imwrite(path, final)


    if __name__ == '__main__':
        make_white_mask()
