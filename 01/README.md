## 练习一

### 1 两数和

题目见https://leetcode-cn.com/problems/two-sum/，可在网页上将编程语言选择为python并提交验证自己的代码检验正确性。

本题答案可提交代码，也可提交代码通过的截图

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]

```

### 2 字符串压缩

题目见https://leetcode-cn.com/problems/compress-string-lcci/，可在网页上将编程语言选择为python并提交验证自己的代码检验正确性。

本题答案可提交代码，也可提交代码通过的截图。

```python
class Solution:
    def compressString(self, S: str) -> str:
        if len(S) <= 1:
            return S
        else:
            C = ""
            i = 0
            count = 1
            for j in range(1, len(S)):
                if j < len(S) - 1:
                    if S[j] == S[i]:
                        count += 1
                    else:
                        C = C + S[i] + str(count)
                        i = j
                        count = 1
                else:
                    if S[j] == S[i]:
                        count += 1
                        C = C + S[i] + str(count)
                    else:
                        C = C + S[i] + str(count) + S[j] + "1"
            if len(C) < len(S):
                return C
            else:
                return S

```

### 3 文件读写

读取附件中的`dec.txt`文件,读取文件各行**十进制数据**`(x,y)`,并转化为**8位二进制补码**,输出到`bin.txt`文件中。

关于补码的概念可以查看百度百科https://baike.baidu.com/item/%E8%A1%A5%E7%A0%81/6854613。简单来说，补码是一种表示正负数的方案。对于正数，补码直接用该数的二进制表示，如`5`表示为`00000101`；对于负数，它的补码应该满足“和正数的补码相加为零”这一条件，比如`-5`表示为`11111001`。

例：

____________________

输入文件格式：

(1,2)

...

(-3,4)

____________________

输出文件格式:

(00000001,00000010)

...

(111111101,00000100)

____________________

提示：可以用`bin()`函数或者模板字符串完成数值到二进制的转换

```python
with open("dec.txt", 'r', encoding='utf8') as f1:
    with open("bin.txt", 'w', encoding='utf8') as f2:
        for i in f1:
            j = eval(i)
            k = [j[0], j[1]]
            for l in range(2):
                if k[l] >= 0:
                    k[l] = int(bin(k[l])[2:])
                else:
                    k[l] = int(bin(256 + k[l])[2:])
            f2.write(f'({"%08d"%k[0]},{"%08d"%k[1]})\n')

```

## 练习二

### 练习2

在本次练习中，我们将生成一个字符图像数据集，以供之后的相似图像识别任务使用。

##### 要求

给定字符集为Unicode值是0x0021\~0x007E、0x00A1\~0x00AC、0x00AE\~0x0377、0x037A\~0x037F、0x0384\~0x038A、0x038C\~0x038C、0x038E\~0x052F的字符，对每个字符生成一张尺寸为48x48的JPEG格式图像（文件后缀名为.jpeg或.jpg），黑底白字，字符居中并大小适宜。示例如下：

![fig](练习2/figure.png)

##### 提示

* 用`chr()`函数获取数值对应的字符，如`chr(48) -> '0'`（反函数是`ord()`，例如`ord('0') -> 48`）
* 可以查询python的图像处理库pillow的文档，了解如何创建空白图像、在图像上绘制文字。https://pillow.readthedocs.io/en/stable/。

##### 参考框架

可以参考如下框架，并填写其中的TODO部分。

```python
from PIL import Image, ImageDraw, ImageFont
import os


# 字符集
chars = [
    range(0x0021,0x007E+1),range(0x00A1,0x00AC+1),range(0x00AE,0x0377+1),range(0x037A,0x037F+1),
    range(0x0384,0x038A+1),range(0x038C,0x038C+1),range(0x038E,0x052F+1)
]
# TODO: 将chars中的range展开为一维数组：[33,34,...]
chars = ...

# 图像尺寸
image_size = 48
# 图像中心
pos = (image_size // 2, ) * 2
# 字体
font = ImageFont.truetype('arial', 32)

# 创建输出文件夹
os.makedirs('data', exist_ok=True)

for char in chars:
    # TODO: 使用Image.new创建图像
    img = ...

    # TODO: 使用ImageDraw绘制文字
    ...

    # TODO: 使用模板字符串写保存文件名
    img.save(f'...')

```

```python
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

```

