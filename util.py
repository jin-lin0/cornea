import numpy as np
import cv2


def show_img(img_show, title='title'):
    if type(img_show) is list:
        cv2.imshow(title, np.hstack(img_show))
    else:
        cv2.imshow(title, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    x = np.linspace(1, 100, 100)
    np.random.shuffle(x)
    print(x)
