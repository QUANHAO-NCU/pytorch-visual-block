def addGaussianNoise2Image(image):
    """
    给图片添加高斯噪声
    """
    pass


def addGaussianNoise2Vector(image):
    """
    给特征向量添加高斯噪声
    """

    pass


def ConvHW2newHW(HWin: list, HWout: list):
    """
    计算卷积时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - k + 2 * p) // s + 1
                tmpW = (Win - k + 2 * p) // s + 1
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


def ConvTransposeHW2newHW(HWin: list, HWout: list):
    """
    计算转置卷积时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - 1) * s - 2 * p + k
                tmpW = (Win - 1) * s - 2 * p + k
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


def PoolHW2newHW(HWin: list, HWout: list):
    """
    计算池化时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - k + 2 * p) // s + 1
                tmpW = (Win - k + 2 * p) // s + 1
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


if __name__ == '__main__':
    print([i for i in ConvHW2newHW([32, 32], [28, 28])])
