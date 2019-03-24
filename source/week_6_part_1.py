# Обучение без учителя
# Кластеризация
# Алгоритм KMeans

import numpy as np
import pandas
from skimage import img_as_float
from skimage.io import imread, imsave
from sklearn.cluster import KMeans
from source import create_answer_file

# получение и преобразование изображения(все значения в интервал от 0 до 1)
image = img_as_float(imread('../data/parrots.jpg'))
w, h, d = image.shape

# матрица объекты-признаки: характеризует каждый пиксель тремя координатами
# - значениями интенсивности в пространстве RGB
rgb_matrix = pandas.DataFrame(np.reshape(image, (w * h, d)), columns=['R', 'G', 'B'])

# метрика PSNR
psnr = lambda image1, image2: 10 * np.log10(1 / (np.mean((image1 - image2) ** 2)))

# нахождение минимального количество кластеров, при котором значение PSNR выше 20
# (можно рассмотреть не более 20 кластеров)
for n_clusters in range(1, 21):
    pixels = rgb_matrix.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)

    # пиксели, отнесенные в один кластер, заполняются средним цветом по кластеру
    means = pixels.groupby('cluster').mean().values
    mean_image = np.reshape(
        [means[c] for c in pixels['cluster'].values], (w, h, d)
    )
    imsave('../images/parrots_means_' + str(n_clusters) + '.jpg', mean_image)

    # пиксели, отнесенные в один кластер, заполняются медианным цветом по кластеру
    medians = pixels.groupby('cluster').median().values
    median_image = np.reshape([medians[c] for c in pixels['cluster'].values], (w, h, d))
    imsave('../images/parrots_medians_' + str(n_clusters) + '.jpg', median_image)

    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)

    if psnr_mean > 20 or psnr_median > 20:
        create_answer_file('w6_1.txt', str(n_clusters))
        break
