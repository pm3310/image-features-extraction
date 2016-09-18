from PIL import Image
from sklearn.metrics import euclidean_distances

from image_net.model import ImageNetModel

"""
Here is an example of using the ImageNet model to extract features from food images
and find the most similar one from a visual point of view.
"""


def main():
    image_net_model = ImageNetModel()
    reference_chicken_image = Image.open('chicken.jpg')
    reference_chicken_extracted_features = image_net_model.extract_features(reference_chicken_image)

    chicken_2_image = Image.open('chicken_2.jpg')
    chicken_2_extracted_features = image_net_model.extract_features(chicken_2_image)
    dist = euclidean_distances([reference_chicken_extracted_features], [chicken_2_extracted_features])[0][0]
    print("Euclidean distance between reference chicken and chicken 2: %s" % str(dist))

    avocado_image = Image.open('avocado.jpeg')
    avocado_extracted_features = image_net_model.extract_features(avocado_image)
    dist = euclidean_distances([reference_chicken_extracted_features], [avocado_extracted_features])[0][0]
    print("Euclidean distance between reference chicken and avocado: %s" % str(dist))

    avocado_tesco_image = Image.open('avocado_tesco.jpg')
    avocado_tesco_extracted_features = image_net_model.extract_features(avocado_tesco_image)
    dist = euclidean_distances([reference_chicken_extracted_features], [avocado_tesco_extracted_features])[0][0]
    print("Euclidean distance between reference chicken and avocado tesco: %s" % str(dist))


if __name__ == '__main__':
    main()
