import numpy as np
import cairo
import PIL

from PIL import Image, ImageOps


class Synthetic:
    def __init__(self, MAX_X=28, MAX_Y=28, border=4):
        self.MAX_X = MAX_X
        self.MAX_Y = MAX_Y
        self.border = border

    def random_line(self, ctx):
        x = []
        y = []
        for it in range(2):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))

        return self.line(ctx, x, y)

    def line(self, ctx, x, y):
        ctx.move_to(x[0], y[0])
        ctx.line_to(x[1], y[1])

        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)

        ctx.stroke()
        ctx.close_path()
        if (x[0], y[0]) < (x[1], y[1]):
            return x[0], y[0], x[1], y[1]
        else:
            return x[1], y[1], x[0], y[0]

    def get_image(self, img_path='../data/Synthetic/', name='1', line_count=2):
        surface, ctx = self._setup_context(img_path + name + "svgfile1.svg")

        mass = []
        for it in range(line_count):
            width = np.random.randint(1, 6)
            ctx.set_line_width(width)
            ctx.set_source_rgba(1, 1, 1, 1)
            mass.append(self.random_line(ctx) + (width,))
        mass.sort()
        ctx.set_operator(cairo.OPERATOR_MULTIPLY)
        surface.write_to_png(img_path + name + '_nh_gt.png')
        image = Image.open(img_path + name + '_nh_gt.png')
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image = inverted_image.convert('L')
        inverted_image.save(img_path + name + '_nh_gt.png')

        return inverted_image, mass

    def r_line(self, ctx, ln):
        ctx.new_sub_path()
        ctx.save()
        ctx.move_to(ln[0], ln[1])
        ctx.line_to(ln[2], ln[3])

        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        ctx.stroke()
        ctx.close_path()
        ctx.restore()

    def _setup_context(self, path):
        surface = cairo.SVGSurface(path, self.MAX_X, self.MAX_Y)
        ctx = cairo.Context(surface)
        ctx.save()
        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()
        ctx.restore()
        ctx.move_to(0, 0)

        return surface, ctx

    def render(self, lines, img_path='../data/Synthetic/', name='1'):
        surface, ctx = self._setup_context(img_path + name + "svgfile1.svg")

        for ln in lines:
            if ln[-1] < 0.5:
                continue
            width = ln[4]
            ctx.set_line_width(width)
            ctx.set_source_rgba(1, 1, 1, 1)

            self.r_line(ctx, ln)

        ctx.set_operator(cairo.OPERATOR_MULTIPLY)
        surface.write_to_png(img_path + name + '_nh_gt.png')
        image = Image.open(img_path + name + '_nh_gt.png')
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image = inverted_image.convert('L')
        inverted_image.save(img_path + name + '_nh_gt.png')

        return inverted_image