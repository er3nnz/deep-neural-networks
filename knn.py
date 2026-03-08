import numpy as np
import os
import cv2

def load_data(base_path, limit=1000):

    X = []
    Y = []

    classes = sorted(os.listdir(base_path))

    print("Veriler okunuyor...")

    for label, class_name in enumerate(classes):

        class_path = os.path.join(base_path, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for img_name in os.listdir(class_path):

            if count >= limit:
                break

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (32,32))

            X.append(img.flatten())
            Y.append(label)

            count += 1

    X = np.array(X)
    Y = np.array(Y)

    print("Toplam veri:", len(X))

    return X,Y,classes


def l1_distance(a,b):

    distance = 0

    for i in range(len(a)):
        distance += abs(int(a[i]) - int(b[i]))

    return distance


def l2_distance(a,b):

    distance = 0

    for i in range(len(a)):
        diff = int(a[i]) - int(b[i])
        distance += diff*diff

    return np.sqrt(distance)


def knn_predict(X_train, Y_train, test_img, k, metric):

    distances = []

    for i in range(len(X_train)):

        if metric == "L1":
            d = l1_distance(X_train[i], test_img)
        else:
            d = l2_distance(X_train[i], test_img)

        distances.append((d, Y_train[i]))

    distances.sort(key=lambda x: x[0])

    neighbors = distances[:k]

    votes = {}

    for _,label in neighbors:

        if label not in votes:
            votes[label] = 1
        else:
            votes[label] += 1

    predicted = max(votes, key=votes.get)

    return predicted

# veri setinin dosya yolunu kendinize uygun şekilde veriniz.
dataset_path = "cifar10/cifar10/train"

X,Y,class_names = load_data(dataset_path)

print("\nMesafe türü seçin (1 Veya 2)")
print("1 - L1 (Manhattan)")
print("2 - L2 (Euclidean)")

choice = input("Seçim: ")

if choice == "1":
    metric = "L1"
else:
    metric = "L2"

k = int(input("k değeri girin: "))

test_image_path = input("Test görüntüsü yolu: ")

test_img = cv2.imread(test_image_path)

if test_img is None:
    print("Resim okunamadı. Lütfen doğru bir image path veriniz.")
    exit()

test_img = cv2.resize(test_img,(32,32))
test_img = test_img.flatten()

pred = knn_predict(X,Y,test_img,k,metric)

print("\nTahmin edilen sınıf:", class_names[pred])