from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from models.ViT import ViT

mnist_classes = [str(ii) for ii in range(10)]
fruit_classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
                 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
                 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado ripe',
                 'Avocado', 'Banana Lady Finger', 'Banana Red', 'Banana', 'Beetroot', 'Blueberry', 'Cactus fruit',
                 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier',
                 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos',
                 'Corn Husk', 'Corn', 'Cucumber Ripe 2', 'Cucumber Ripe', 'Dates', 'Eggplant', 'Fig', 'Ginger Root',
                 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White 2', 'Grape White 3', 'Grape White 4',
                 'Grape White', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki',
                 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon Meyer', 'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango Red',
                 'Mango', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine Flat', 'Nectarine',
                 'Nut Forest', 'Nut Pecan', 'Onion Red Peeled', 'Onion Red', 'Onion White', 'Orange', 'Papaya',
                 'Passion Fruit', 'Peach 2', 'Peach Flat', 'Peach', 'Pear 2', 'Pear Abate', 'Pear Forelle',
                 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pear', 'Pepino',
                 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis with Husk', 'Physalis',
                 'Pineapple Mini', 'Pineapple', 'Pitahaya Red', 'Plum 2']


def main():
    # root_path = ''

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--image-path', type=str)
    args = parser.parse_args()

    assert args.dataset in ['fruit', 'mnist']

    if args.dataset == 'mnist':
        model = ViT.load_from_checkpoint(
            args.weights,
            image_size=28,
            patch_size=7,
            num_channels=1,
            num_classes=len(mnist_classes),
            d_model=128,
            num_blocks=6,
            num_heads=8,
            mvp_head=512,
            dropout=0.1,
            displ_attention=True,
        )
    else:
        model = ViT.load_from_checkpoint(
            args.weights,
            image_size=100,
            patch_size=10,
            num_channels=3,
            num_classes=len(fruit_classes),
            d_model=128,
            num_blocks=6,
            num_heads=8,
            mvp_head=512,
            dropout=0.1,
            displ_attention=True,
        )

    image = Image.open(args.image_path)
    image_normalized = transforms.ToTensor()(image).unsqueeze(0)
    # print(f"image shape: {image_normalized.size()}")

    model.eval()
    logits = model(image_normalized)
    pred = torch.argmax(logits, dim=-1).item()

    pred_label = mnist_classes[pred] if args.dataset == 'mnist' else fruit_classes[pred]

    print(f"Predicted label: {pred_label}")


if __name__ == '__main__':
    main()
