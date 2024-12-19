import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # veya 'Qt5Agg', 'Agg' gibi alternatif backendler deneyebilirsiniz

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.exposure import histogram
import skfuzzy as fuzz
from skfuzzy import control as ctrl

############################################################
# AŞAMA 1: Veri Setinin Yüklenmesi
############################################################

data_dir = r"C:\Users\hasan\Desktop\archive\Blood_Cancer1"

image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.tiff')]

if len(image_paths) == 0:
    raise FileNotFoundError("Belirtilen klasörde .tiff formatında görüntü bulunamadı!")

############################################################
# AŞAMA 2: Özellik Çıkarma Fonksiyonları
############################################################

def calc_brightness(img):
    if img.shape[-1] == 4:
        img = img[..., :3]
    gray = rgb2gray(img)
    return np.mean(gray)

def calc_contrast(img):
    if img.shape[-1] == 4:
        img = img[..., :3]
    gray = rgb2gray(img)
    return np.std(gray)

def calc_sharpness(img):
    if img.shape[-1] == 4:
        img = img[..., :3]
    gray = rgb2gray(img)
    lap_var = laplace(gray).var()
    return lap_var

def calc_noise(img):
    if img.shape[-1] == 4:
        img = img[..., :3]
    gray = rgb2gray(img)
    hist_values, hist_centers = histogram(gray)
    noise_metric = np.std(hist_values)
    return noise_metric

def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin + 1e-9)

############################################################
# AŞAMA 3: Tüm Görüntüler İçin Özelliklerin Hesaplanması
############################################################

features = []
images = []  # Görüntüleri sonradan görselleştirmek için tutalım
for path in image_paths:
    img = imread(path)
    b = calc_brightness(img)
    c = calc_contrast(img)
    s = calc_sharpness(img)
    n = calc_noise(img)
    features.append([path, s, c, n, b])
    images.append((path, img))

df = pd.DataFrame(features, columns=["path", "sharpness", "contrast", "noise", "brightness"])

############################################################
# AŞAMA 4: Normalizasyon
############################################################

s_min, s_max = df['sharpness'].min(), df['sharpness'].max()
c_min, c_max = df['contrast'].min(), df['contrast'].max()
n_min, n_max = df['noise'].min(), df['noise'].max()
b_min, b_max = df['brightness'].min(), df['brightness'].max()

df['s_norm'] = df['sharpness'].apply(lambda x: normalize(x, s_min, s_max))
df['c_norm'] = df['contrast'].apply(lambda x: normalize(x, c_min, c_max))
df['n_norm'] = df['noise'].apply(lambda x: normalize(x, n_min, n_max))
df['b_norm'] = df['brightness'].apply(lambda x: normalize(x, b_min, b_max))

############################################################
# AŞAMA 5: Bulanık Mantık Sistemi Tanımlama (Kalite)
############################################################

sharpness_var = ctrl.Antecedent(np.linspace(0, 1, 100), 'sharpness')
contrast_var = ctrl.Antecedent(np.linspace(0, 1, 100), 'contrast')
noise_var = ctrl.Antecedent(np.linspace(0, 1, 100), 'noise')
brightness_var = ctrl.Antecedent(np.linspace(0, 1, 100), 'brightness')

quality = ctrl.Consequent(np.linspace(0, 1, 100), 'quality')

sharpness_var['low'] = fuzz.trimf(sharpness_var.universe, [0, 0, 0.4])
sharpness_var['medium'] = fuzz.trimf(sharpness_var.universe, [0.3, 0.5, 0.7])
sharpness_var['high'] = fuzz.trimf(sharpness_var.universe, [0.6, 1, 1])

contrast_var['low'] = fuzz.trimf(contrast_var.universe, [0, 0, 0.4])
contrast_var['medium'] = fuzz.trimf(contrast_var.universe, [0.3, 0.5, 0.7])
contrast_var['high'] = fuzz.trimf(contrast_var.universe, [0.6, 1, 1])

noise_var['low'] = fuzz.trimf(noise_var.universe, [0, 0, 0.4])
noise_var['medium'] = fuzz.trimf(noise_var.universe, [0.3, 0.5, 0.7])
noise_var['high'] = fuzz.trimf(noise_var.universe, [0.6, 1, 1])

brightness_var['low'] = fuzz.trimf(brightness_var.universe, [0, 0, 0.4])
brightness_var['medium'] = fuzz.trimf(brightness_var.universe, [0.3, 0.5, 0.7])
brightness_var['high'] = fuzz.trimf(brightness_var.universe, [0.6, 1, 1])

quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 0.3])
quality['medium'] = fuzz.trimf(quality.universe, [0.2, 0.5, 0.8])
quality['excellent'] = fuzz.trimf(quality.universe, [0.7, 1, 1])

rule1 = ctrl.Rule(sharpness_var['high'] & contrast_var['high'] & noise_var['low'], quality['excellent'])
rule2 = ctrl.Rule(sharpness_var['low'] | noise_var['high'], quality['poor'])
rule3 = ctrl.Rule(contrast_var['medium'] & brightness_var['medium'] & noise_var['medium'], quality['medium'])
rule4 = ctrl.Rule(sharpness_var['medium'] & contrast_var['high'] & noise_var['low'], quality['excellent'])

rule_default = ctrl.Rule(brightness_var['low'] | brightness_var['medium'] | brightness_var['high'], quality['medium'])

quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule_default])
quality_eval = ctrl.ControlSystemSimulation(quality_ctrl)

############################################################
# AŞAMA 6: Kalite Skorlarının Hesaplanması
############################################################

quality_scores = []
for i, row in df.iterrows():
    quality_eval.input['sharpness'] = row['s_norm']
    quality_eval.input['contrast'] = row['c_norm']
    quality_eval.input['noise'] = row['n_norm']
    quality_eval.input['brightness'] = row['b_norm']
    quality_eval.compute()
    q = quality_eval.output['quality']
    quality_scores.append(q)

df['quality_score'] = quality_scores

############################################################
# AŞAMA 7: Sonuçların İncelenmesi
############################################################

print("İlk 5 Sonuç:")
print(df.head())

plt.hist(df['quality_score'], bins=10, edgecolor='black')
plt.title("Quality Score Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

plt.scatter(df['sharpness'], df['quality_score'])
plt.title("Sharpness vs Quality Score")
plt.xlabel("Sharpness (Ham Değer)")
plt.ylabel("Quality Score")
plt.show()

plt.hist(df['sharpness'], bins=10, edgecolor='black', color='blue')
plt.title("Distribution of Sharpness")
plt.xlabel("Sharpness")
plt.ylabel("Count")
plt.show()

plt.hist(df['contrast'], bins=10, edgecolor='black', color='green')
plt.title("Distribution of Contrast")
plt.xlabel("Contrast")
plt.ylabel("Count")
plt.show()

plt.hist(df['noise'], bins=10, edgecolor='black', color='red')
plt.title("Distribution of Noise Level")
plt.xlabel("Noise Level")
plt.ylabel("Count")
plt.show()

plt.hist(df['brightness'], bins=10, edgecolor='black', color='purple')
plt.title("Distribution of Brightness")
plt.xlabel("Brightness")
plt.ylabel("Count")
plt.show()

############################################################
# AŞAMA 8: Bulanık Eşikleme ile Segmentasyon
############################################################

# Burada pikselin gri değerine göre bulanık üyelik fonksiyonları tanımlıyoruz.
# Amaç: Pikselin 'foreground' (nesne) veya 'background' (arka plan) üyelik derecelerini bulanık mantıkla belirlemek.

intensity = ctrl.Antecedent(np.linspace(0,1,100), 'intensity')
# intensity için üyelik fonksiyonları
intensity['dark'] = fuzz.trapmf(intensity.universe, [0,0,0.3,0.5])
intensity['medium'] = fuzz.trimf(intensity.universe, [0.3,0.5,0.7])
intensity['bright'] = fuzz.trapmf(intensity.universe, [0.5,0.7,1,1])

# Çıkış: membership to foreground (nesne)
foreground = ctrl.Consequent(np.linspace(0,1,100), 'foreground')
foreground['low'] = fuzz.trimf(foreground.universe, [0,0,0.5])
foreground['high'] = fuzz.trimf(foreground.universe, [0.5,1,1])

# Kurallar:
# Eğer piksel dark ise foreground düşük
rule_f1 = ctrl.Rule(intensity['dark'], foreground['low'])
# Eğer piksel bright ise foreground yüksek
rule_f2 = ctrl.Rule(intensity['bright'], foreground['high'])
# Eğer piksel medium ise foreground orta seviyede olsun,
# basitçe hem high hem low arası bir kural ekleyebiliriz:
rule_f3 = ctrl.Rule(intensity['medium'], foreground['high'])

foreground_ctrl = ctrl.ControlSystem([rule_f1, rule_f2, rule_f3])
foreground_eval = ctrl.ControlSystemSimulation(foreground_ctrl)

def fuzzy_thresholding(img):
    gray = rgb2gray(img)
    # Her piksel için bulanık çıkarım yap
    out = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            val = gray[i,j]
            foreground_eval.input['intensity'] = val
            foreground_eval.compute()
            fg_val = foreground_eval.output['foreground']
            out[i,j] = fg_val
    # Şimdi out 0-1 arası değerler içeriyor
    # Bir eşik seçelim. Örneğin 0.5 üzeri foreground kabul edelim.
    binary_mask = out > 0.5
    return binary_mask

def show_images_with_fuzzy_threshold(img_list):
    for path, img in img_list:
        if img.shape[-1] == 4:
            img = img[..., :3]

        row = df[df['path'] == path].iloc[0]
        q_score = row['quality_score']

        # Bulanık Eşikleme ile segmentasyon
        binary_mask = fuzzy_thresholding(img)

        mask_colored = np.zeros_like(img)
        mask_colored[binary_mask] = [255, 0, 0]
        combined = (0.5 * img + 0.5 * mask_colored).astype(np.uint8)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title(f"Görüntü: {os.path.basename(path)}\nKalite Skoru: {q_score:.4f}")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Bulanık Eşikleme Maskesi")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(combined)
        plt.title("Maske + Orijinal")
        plt.axis('off')

        plt.show()

print("İlk 3 Görsel")
first_3 = images[:3]
show_images_with_fuzzy_threshold(first_3)

print("Son 3 Görsel")
last_3 = images[-3:]
show_images_with_fuzzy_threshold(last_3)
