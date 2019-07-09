import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import load_model


LABELS = ["Jeanne", "JeanneAlt", "JeanneAltLi"]

#保存したモデル

model = load_model('jeanne_model.hdf5')

def check_jeanne(url):
    # 対象画像のインポート
    img = Image.open(url)
    img = img.convert("RGB")
    img = img.resize((256, 256))
    plt.imshow(img)
    plt.show()

    x = np.asarray(img)
    x = x.reshape(-1, 256, 256, 3)
    x = x / 255

    # 予測
    pre = model.predict(x)
    idx = np.argmax(pre, axis=1)[0]
    per = int(pre[0][idx] * 100)
    return (idx, per, img)

def check_jeanne_result(url):
    idx, per, img = check_jeanne(url)
    #答えを表示
    print("この写真は、", LABELS[idx])
    print(per, "%の可能性で合っているだろう")
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    check_jeanne_result('train/Jeanne/IMG_0409.JPG')
    check_jeanne_result('train/JeanneAlt/IMG_0509.JPG')
    check_jeanne_result('test/JeanneAltLi/IMG_0603.JPG')
    print(model.summary())
