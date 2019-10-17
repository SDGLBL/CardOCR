from PIL import ImageDraw, Image


# 通过box生成画线的图像
def draw_img_by_box(img, box, rgb=(208, 100, 84), width=2, savepath=None):
    img = Image.fromarray(img)
    draw_board = ImageDraw.Draw(img)
    for index, vetor in enumerate(box):
        x1, y1, x2, y2, x3, y3, x4, y4 = vetor[:8]
        draw_board.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)], rgb, width)
    if savepath != None:
        img.save(savepath)
    return img
