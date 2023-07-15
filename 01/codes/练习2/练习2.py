from PIL import Image, ImageDraw, ImageFont
import os


# 字符集
chars = [
    range(0x0021, 0x007E+1), range(0x00A1, 0x00AC +
                                   1), range(0x00AE, 0x0377+1), range(0x037A, 0x037F+1),
    range(0x0384, 0x038A+1), range(0x038C, 0x038C+1), range(0x038E, 0x052F+1)
]
# 将chars中的range展开为一维数组：[33,34,...]
chars_ = []
for i in chars:
    for j in i:
        chars_.append(j)
chars = chars_

# 图像尺寸
image_size = 48
# 图像中心
pos = (image_size // 2, ) * 2
# 字体
fnt = ImageFont.truetype('Noto Sans Regular', 32)

# 创建输出文件夹
os.makedirs('data', exist_ok=True)

for char in chars:
    # 使用Image.new创建图像
    img = Image.new('RGB', [image_size, image_size], color=(0, 0, 0))

    # 使用ImageDraw绘制文字
    draw = ImageDraw.Draw(img)
    draw.text(pos, chr(char), fill=(255, 255, 255), font=fnt, anchor="mm")

    # 使用模板字符串写保存文件名
    img.save(f'./data/0x{"%04X"%char}.jpg', 'JPEG')
